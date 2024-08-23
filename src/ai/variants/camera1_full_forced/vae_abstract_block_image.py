import torch.optim as optim
import numpy as np

from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.modules.save_load_handlers.ai_models_handle import save_ai_manually, load_manually_saved_ai
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from typing import List, Tuple
from src.utils import array_to_tensor
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, pretty_display_set
from src.ai.variants.tests.eval_vae_abstract_block import evaluate_confidence_vae_abstract, \
    evaluate_reconstruct_vae_abstract, evaluate_adjacency_properties_vae_abstract, \
    evaluate_distances_between_pairs_vae_abstract

import torch
import torch.nn as nn
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer_no_batchnorm_leaky, _make_layer_linear


def reparameterization(mean, var):
    epsilon = torch.randn_like(var)
    z = mean + var * epsilon  # reparameterization trick
    return z


class VAEAutoencoderAbstractionBlockImage(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, position_embedding_size: int = 48, thetas_embedding_size: int = 48,
                 hidden_size: int = 256, num_blocks: int = 1, input_output_size=512,
                 concatenated_instances: int = 1):
        super(VAEAutoencoderAbstractionBlockImage, self).__init__()
        self.concatenated_instances = concatenated_instances
        input_output_size *= concatenated_instances
        self.input_layer = nn.Linear(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        # Separate encoders for mean and log variance
        self.position_mean_encoder = _make_layer_no_batchnorm_leaky(hidden_size, position_embedding_size)
        self.position_logvar_encoder = _make_layer_linear(hidden_size, position_embedding_size)
        self.thetas_mean_encoder = _make_layer_no_batchnorm_leaky(hidden_size, thetas_embedding_size)
        self.thetas_logvar_encoder = _make_layer_linear(hidden_size, thetas_embedding_size)

        self.decoder_initial_layer = _make_layer_no_batchnorm_leaky(position_embedding_size + thetas_embedding_size,
                                                                    hidden_size)
        self.decoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, input_output_size)

        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.embedding_size = position_embedding_size

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder_training(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = self.leaky_relu(x)
        for block in self.encoding_blocks:
            x = block(x)

        position_mean = self.position_mean_encoder(x)
        position_logvar = self.position_logvar_encoder(x)

        thetas_mean = self.thetas_mean_encoder(x)
        thetas_logvar = self.thetas_logvar_encoder(x)

        return position_mean, position_logvar, thetas_mean, thetas_logvar

    def encoder_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        position_mean, position_logvar, thetas_mean, thetas_logvar = self.encoder_training(x)
        return position_mean, thetas_mean

    def decoder_training(self, position: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        thetas *= 0
        x = torch.cat([position, thetas], dim=-1)
        x = self.decoder_initial_layer(x)
        x = self.leaky_relu(x)
        for block in self.decoding_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

    def decoder_inference(self, position: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(position, thetas)

    def forward_training(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        position_mean, position_logvar, thetas_mean, thetas_logvar = self.encoder_training(x)
        position = self.reparameterize(position_mean, position_logvar)
        thetas = self.reparameterize(thetas_mean, thetas_logvar)
        reconstruction = self.decoder_training(position, thetas)
        return reconstruction, position_mean, position_logvar, thetas_mean, thetas_logvar

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        position, thetas = self.encoder_inference(x)
        return self.decoder_inference(position, thetas)

    def get_embedding_size(self) -> int:
        return self.embedding_size


def vae_loss(x: torch.Tensor, x_recon: torch.Tensor, position_mean: torch.Tensor, position_logvar: torch.Tensor,
             thetas_mean: torch.Tensor, thetas_logvar: torch.Tensor, beta: float = 1.0) -> Tuple[
    torch.Tensor, torch.Tensor]:
    criterion_mse = nn.MSELoss()
    recon_loss = criterion_mse(x_recon, x)

    position_kld = -0.5 * torch.mean(1 + position_logvar - position_mean.pow(2) - position_logvar.exp())
    thetas_kld = -0.5 * torch.mean(1 + thetas_logvar - thetas_mean.pow(2) - thetas_logvar.exp())

    return recon_loss, beta * (position_kld + thetas_kld)


def reconstruction_handling(autoencoder: BaseAutoencoderModel, data: any,
                            scale_reconstruction_loss: float = 1, beta: float = 0):
    position_mean, position_logvar, theta_mean, theta_logvar = autoencoder.encoder_training(data)
    position = position_mean
    theta = theta_mean
    position = autoencoder.reparameterize(position_mean, position_logvar)
    theta = autoencoder.reparameterize(theta_mean, theta_logvar)
    reconstruction = autoencoder.decoder_training(position, theta)

    reconstruction_loss, KL_loss = vae_loss(data, reconstruction, position_mean, position_logvar, theta_mean,
                                            theta_logvar, beta=beta)

    return reconstruction_loss * scale_reconstruction_loss, KL_loss


def adjacent_distance_handling(autoencoder: BaseAutoencoderModel, adjacent_sample_size: int,
                               scale_adjacent_distance_loss: float) -> tuple[torch.Tensor, float]:
    """
    Keeps adjacent pairs close to each other
    """
    sampled_pairs = storage_raw.sample_adjacent_datapoints_connections(adjacent_sample_size)

    adjacent_distance_loss = torch.tensor(0.0)
    average_distance = 0
    batch_datapoint1 = []
    batch_datapoint2 = []
    for pair in sampled_pairs:
        # keep adjacent close to each other
        data_point1 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        data_point2 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["end"])
        batch_datapoint1.append(data_point1)
        batch_datapoint2.append(data_point2)

    batch_datapoint1 = torch.stack(batch_datapoint1)
    batch_datapoint2 = torch.stack(batch_datapoint2)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    # embedding_size = encoded_i.shape[1]

    distance = torch.sum(torch.norm((encoded_i - encoded_j), p=2))

    average_distance += distance.item() / adjacent_sample_size
    adjacent_distance_loss += distance / adjacent_sample_size * scale_adjacent_distance_loss

    return adjacent_distance_loss, average_distance


def non_adjacent_distance_handling(autoencoder: BaseAutoencoderModel, non_adjacent_sample_size: int,
                                   scale_non_adjacent_distance_loss: float, distance_per_neuron: float) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other
    """
    sampled_pairs = storage_raw.sample_datapoints_adjacencies_cheated(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []

    for pair in sampled_pairs:
        datapoint1 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        datapoint2 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["end"])

        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1)
    batch_datapoint2 = torch.stack(batch_datapoint2)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] * distance_per_neuron * embedding_size for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance)

    criterion = nn.MSELoss()
    non_adjacent_distance_loss = criterion(distance, expected_distance) * scale_non_adjacent_distance_loss
    return non_adjacent_distance_loss


def permutation_adjustion_handling(autoencoder: BaseAutoencoderModel, samples: int,
                                   scale_permutation_adjustion_loss: float) -> torch.Tensor:
    """
    Keeps the permutation of the data points close to each other
    """
    global storage_raw

    datapoint: List[str] = storage.sample_n_random_datapoints(samples)
    datapoints_data = [storage.get_datapoint_data_tensor_by_name(name) for name in datapoint]
    accumulated_loss = torch.tensor(0.0)
    for datapoint_data in datapoints_data:
        enc = autoencoder.encoder_training(datapoint_data)
        loss = torch.cdist(enc, enc, p=2).mean()
        accumulated_loss += loss

    return accumulated_loss / samples * scale_permutation_adjustion_loss


recons_data = None


def reconstruction_handling_with_freezing(autoencoder: BaseAutoencoderModel,
                                          scale_reconstruction_loss: float = 1) -> any:
    global storage_raw, recons_data
    sampled_count = 25
    sampled_points = storage.sample_n_random_datapoints(sampled_count)

    loss_reconstruction = torch.tensor(0.0, device=device)
    loss_freezing_same_position = torch.tensor(0.0, device=device)
    loss_freezing_same_rotation = torch.tensor(0.0, device=device)

    if recons_data is None:
        recons_data = []
        for point in sampled_points:
            for i in range(OFFSETS_PER_DATAPOINT):
                sampled_full_rotation = array_to_tensor(
                    storage.get_point_rotations_with_full_info_set_offset_concatenated(point, ROTATIONS_PER_FULL, i))
                recons_data.append(sampled_full_rotation)

    data = torch.stack(recons_data).to(device=device)

    positional_encoding, thetas_encoding = autoencoder.encoder_inference(data)
    dec = autoencoder.decoder_training(positional_encoding, thetas_encoding)

    criterion_reconstruction = nn.MSELoss()

    loss_reconstruction = criterion_reconstruction(dec, data)
    # traverses each OFFSET_PER_DATAPOINT batch, selects tensors from each batch and calculates the loss

    position_encoding_change_on_position = torch.tensor(0.0, device=device)
    position_encoding_change_on_rotation = torch.tensor(0.0, device=device)

    rotation_encoding_change_on_position = torch.tensor(0.0, device=device)
    rotation_encoding_change_on_rotation = torch.tensor(0.0, device=device)

    # rotation changes, position stays the same
    for i in range(sampled_count):
        start_index = i * OFFSETS_PER_DATAPOINT
        end_index = (i + 1) * OFFSETS_PER_DATAPOINT
        positional_encs = positional_encoding[start_index:end_index]
        rotational_encs = thetas_encoding[start_index:end_index]

        loss_freezing_same_position += torch.cdist(positional_encs, positional_encs).mean()

        position_encoding_change_on_rotation += torch.cdist(positional_encs, positional_encs).mean()
        rotation_encoding_change_on_rotation += torch.cdist(rotational_encs, rotational_encs).mean()

    loss_freezing_same_position /= sampled_count

    position_encoding_change_on_rotation /= sampled_count
    rotation_encoding_change_on_rotation /= sampled_count

    rotation_constant_array_rotation_embeddings = []
    rotation_constant_array_position_embeddings = []

    # putting same rotations in a list one after another
    for rotation_offset in range(OFFSETS_PER_DATAPOINT):
        for position_index in range(sampled_count):
            idx = position_index * OFFSETS_PER_DATAPOINT + rotation_offset
            rotation_constant_array_rotation_embeddings.append(thetas_encoding[idx])
            rotation_constant_array_position_embeddings.append(positional_encoding[idx])

    rotation_constant_array_rotation_embeddings = torch.stack(rotation_constant_array_rotation_embeddings).to(
        device=device)
    rotation_constant_array_position_embeddings = torch.stack(rotation_constant_array_position_embeddings).to(
        device=device
    )

    # position changes, rotation stays the same
    for i in range(OFFSETS_PER_DATAPOINT):
        start_index = i * sampled_count
        end_index = (i + 1) * sampled_count

        rotational_encs = rotation_constant_array_rotation_embeddings[start_index:end_index]
        positional_encs = rotation_constant_array_position_embeddings[start_index:end_index]

        loss_freezing_same_rotation += torch.cdist(rotational_encs, rotational_encs).mean()

        position_encoding_change_on_position += torch.cdist(positional_encs, positional_encs).mean()
        rotation_encoding_change_on_position += torch.cdist(rotational_encs, rotational_encs).mean()

    loss_freezing_same_rotation /= sampled_count

    position_encoding_change_on_position /= sampled_count
    rotation_encoding_change_on_position /= sampled_count

    loss_reconstruction *= scale_reconstruction_loss
    loss_freezing_same_position *= 1
    loss_freezing_same_rotation *= 1

    ratio_loss_position = position_encoding_change_on_rotation / position_encoding_change_on_position
    ratio_loss_rotation = rotation_encoding_change_on_position / rotation_encoding_change_on_rotation

    return {
        "loss_reconstruction": loss_reconstruction,
        "loss_freezing_same_position": loss_freezing_same_position,
        "loss_freezing_same_rotation": loss_freezing_same_rotation,
        "ratio_loss_position": ratio_loss_position,
        "ratio_loss_rotation": ratio_loss_rotation,
        "rotation_encoding_change_on_position": rotation_encoding_change_on_position,
        "rotation_encoding_change_on_rotation": rotation_encoding_change_on_rotation,
        "position_encoding_change_on_position": position_encoding_change_on_position,
        "position_encoding_change_on_rotation": position_encoding_change_on_rotation
    }


def train_vae_abstract_block(autoencoder: BaseAutoencoderModel, epochs: int,
                             pretty_print: bool = False) -> BaseAutoencoderModel:
    # PARAMETERS
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005, amsgrad=True)
    autoencoder = autoencoder.to(device)

    num_epochs = epochs

    epoch_average_loss = 0
    reconstruction_average_loss = 0
    loss_same_position_average_loss = 0
    loss_same_rotation_average_loss = 0
    loss_ratio_position_average_loss = 0
    loss_ratio_rotation_average_loss = 0

    epoch_print_rate = 1000

    rotation_encoding_change_on_position_avg = 0
    rotation_encoding_change_on_rotation_avg = 0

    position_encoding_change_on_position_avg = 0
    position_encoding_change_on_rotation_avg = 0

    storage_raw.build_permuted_data_random_rotations_rotation0()
    train_data = array_to_tensor(np.array(storage_raw.get_pure_permuted_raw_env_data())).to(device)

    SHUFFLE_RATE = 5
    beta = 0
    scale_reconstruction_loss = 1

    if pretty_print:
        pretty_display_set(epoch_print_rate, "Epoch batch")
        pretty_display_start(0)

    for epoch in range(num_epochs):
        if (epoch % SHUFFLE_RATE == 0):
            # storage.build_permuted_data_random_rotations()
            # train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(device)
            pass

        reconstruction_loss = torch.tensor(0.0)
        loss_same_position = torch.tensor(0.0)
        loss_same_rotation = torch.tensor(0.0)
        ratio_loss_position = torch.tensor(0.0)
        ratio_loss_rotation = torch.tensor(0.0)

        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        reconstruction_loss, KL_loss = reconstruction_handling(autoencoder, train_data, scale_reconstruction_loss, beta)

        # losses_json = reconstruction_handling_with_freezing(
        #     autoencoder,
        #     scale_reconstruction_loss)
        # ratio_loss_position = losses_json["ratio_loss_position"]
        # ratio_loss_rotation = losses_json["ratio_loss_rotation"]
        # loss_same_position = losses_json["loss_freezing_same_position"]
        # loss_same_rotation = losses_json["loss_freezing_same_rotation"]
        #
        # rotation_encoding_change_on_position = losses_json["rotation_encoding_change_on_position"]
        # rotation_encoding_change_on_rotation = losses_json["rotation_encoding_change_on_rotation"]
        # position_encoding_change_on_position = losses_json["position_encoding_change_on_position"]
        # position_encoding_change_on_rotation = losses_json["position_encoding_change_on_rotation"]
        #
        # loss_same_position.backward(retain_graph=True)
        # loss_same_rotation.backward(retain_graph=True)
        # ratio_loss_rotation.backward(retain_graph=True)
        # ratio_loss_position.backward(retain_graph=True)

        reconstruction_loss.backward(retain_graph=True)
        KL_loss.backward()

        optimizer.step()

        if pretty_print:
            pretty_display(epoch % epoch_print_rate)

        epoch_loss += reconstruction_loss.item() + loss_same_position.item() + loss_same_rotation.item() + ratio_loss_position.item() + ratio_loss_rotation.item() + KL_loss.item()
        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()

        if epoch % epoch_print_rate == 0 and epoch != 0:

            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            loss_same_position_average_loss /= epoch_print_rate
            loss_same_rotation_average_loss /= epoch_print_rate
            loss_ratio_position_average_loss /= epoch_print_rate
            loss_ratio_rotation_average_loss /= epoch_print_rate

            rotation_encoding_change_on_position_avg /= epoch_print_rate
            rotation_encoding_change_on_rotation_avg /= epoch_print_rate

            position_encoding_change_on_position_avg /= epoch_print_rate
            position_encoding_change_on_rotation_avg /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs}")
            # print(f"average distance between adjacent: {average_distance_adjacent}")

            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | LOSS SAME POSITION:{loss_same_position_average_loss} | LOSS SAME ROTATION:{loss_same_rotation_average_loss} | RATIO LOSS POSITION:{loss_ratio_position_average_loss} | RATIO LOSS ROTATION:{loss_ratio_rotation_average_loss}")
            print(
                f"Changes of rotation encoding on position: {rotation_encoding_change_on_position_avg} | Changes of rotation encoding on rotation: {rotation_encoding_change_on_rotation_avg}")
            print(
                f"Changes of position encoding on position: {position_encoding_change_on_position_avg} | Changes of position encoding on rotation: {position_encoding_change_on_rotation_avg}")

            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            loss_same_position_average_loss = 0
            loss_same_rotation_average_loss = 0
            loss_ratio_position_average_loss = 0
            loss_ratio_rotation_average_loss = 0

            if pretty_print:
                pretty_display_reset()
                pretty_display_start(epoch)

    return autoencoder


def run_tests(autoencoder):
    global storage_raw

    evaluate_reconstruct_vae_abstract(autoencoder, storage, rotations0=False)
    evaluate_confidence_vae_abstract(autoencoder, storage, rotations0=False)
    # avg_distance_adj = evaluate_distances_between_pairs_vae_abstract(autoencoder, storage, rotations0=False)
    # evaluate_adjacency_properties_vae_abstract(autoencoder, storage, avg_distance_adj, rotation0=False)


def run_loaded_ai():
    autoencoder = load_manually_saved_ai("camera1_full_forced_saved.pth")
    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = VAEAutoencoderAbstractionBlockImage()
    train_vae_abstract_block(autoencoder, epochs=10001, pretty_print=True)
    save_ai_manually("vae_image_full_forced", autoencoder)
    run_tests(autoencoder)


def run_vae_abstract_block() -> None:
    global storage_raw

    dataset_grid = 5

    storage.load_raw_data_from_others(f"data{dataset_grid}x{dataset_grid}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{dataset_grid}x{dataset_grid}_connections.json")
    # selects first rotation
    storage.build_permuted_data_random_rotations_rotation0()
    run_new_ai()

    # run_loaded_ai()


storage_raw: StorageSuperset2 = StorageSuperset2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROTATIONS_PER_FULL = 1
OFFSETS_PER_DATAPOINT = 24
TOTAL_ROTATIONS = 24
