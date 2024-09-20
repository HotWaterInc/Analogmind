import torch.optim as optim
import numpy as np
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer_no_batchnorm_leaky, _make_layer
from src.navigation_core import BaseAutoencoderModel
from src.navigation_core import ROTATIONS_PER_FULL, OFFSETS_PER_DATAPOINT
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_storages.storage_superset2 import StorageSuperset2
from src.utils import array_to_tensor, get_device
from src.utils.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, pretty_display_set
import torch
import torch.nn as nn


class AbstractionBlockSecondTrial(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, position_embedding_size: int = 512,
                 thetas_embedding_size: int = 512,
                 hidden_size: int = 1024 * 4, num_blocks: int = 1,
                 input_output_size=512 * ROTATIONS_PER_FULL):
        super(AbstractionBlockSecondTrial, self).__init__()

        self.input_layer = _make_layer(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.positional_encoder = _make_layer(hidden_size, position_embedding_size)
        self.thetas_encoder = _make_layer(hidden_size, thetas_embedding_size)

        self.decoder_initial_layer = _make_layer(position_embedding_size + thetas_embedding_size,
                                                 hidden_size)
        self.decoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.output_layer = _make_layer_no_batchnorm_leaky(hidden_size, input_output_size)
        self.embedding_size = position_embedding_size
        self.sigmoid = nn.Sigmoid()

    def encoder_training(self, x: torch.Tensor) -> any:

        x = self.input_layer(x)
        for block in self.encoding_blocks:
            x = block(x)

        position = self.positional_encoder(x)
        thetas = self.thetas_encoder(x)

        return position, thetas

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        position, _ = self.encoder_training(x)
        return position

    def decoder_training(self, positional_encoding, rotational_encoding) -> torch.Tensor:
        x = torch.cat([positional_encoding, rotational_encoding], dim=-1)
        x = self.decoder_initial_layer(x)

        for block in self.decoding_blocks:
            x = block(x)

        x = self.output_layer(x)
        return x

    def decoder_inference(self, positional_encoding, rotational_encoding) -> torch.Tensor:
        return self.decoder_training(positional_encoding, rotational_encoding)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        positional_encoding, rotational_encoding = self.encoder_training(x)
        return self.decoder_training(positional_encoding, rotational_encoding)

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)

    def get_embedding_size(self) -> int:
        return self.embedding_size


_cache_reconstruction_loss = {}
_cache_thetas = {}


def same_position_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2, data_arr: any) -> any:
    data_arr_tensor = torch.stack([torch.stack(pda) for pda in data_arr]).to(get_device())
    # reshape
    original_shape_data = data_arr_tensor.shape
    x_reshaped = data_arr_tensor.view(-1, original_shape_data[2])
    # forward
    positional_encoding, thetas_encoding = autoencoder.encoder_training(x_reshaped)
    # reshape back
    positional_encoding = positional_encoding.view(original_shape_data[0], original_shape_data[1], -1)
    thetas_encoding_reshaped = thetas_encoding.view(original_shape_data[0], original_shape_data[1], -1)

    cdist_positional_encoding_same_position = torch.cdist(positional_encoding, positional_encoding, p=2).mean()
    cdist_rotation_encoding_same_position = torch.cdist(thetas_encoding_reshaped, thetas_encoding_reshaped, p=2).mean()

    # average over each dimension of the embedding
    positional_encoding = positional_encoding.mean(dim=1)
    # duplicate to batch size
    positional_encoding = positional_encoding.to("cpu").detach().numpy()
    positional_encoding = np.repeat(positional_encoding[:, np.newaxis, :], OFFSETS_PER_DATAPOINT, axis=1)
    positional_encoding = array_to_tensor(positional_encoding).to(get_device())
    new_shape_data = positional_encoding.shape

    # reshape again
    positional_encoding = positional_encoding.view(-1, new_shape_data[2])

    dec = autoencoder.decoder_training(positional_encoding, thetas_encoding)
    criterion = nn.MSELoss()
    loss_reconstruction_same_position = criterion(dec, x_reshaped)

    return loss_reconstruction_same_position, cdist_positional_encoding_same_position, cdist_rotation_encoding_same_position


def get_shape(lst):
    if not isinstance(lst, list):
        return []
    return [len(lst)] + get_shape(lst[0])


def same_rotation_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2, data_arr: any) -> any:
    data_arr_same_rotations = []
    # data arr is points x rotations x size
    sampled_count = len(data_arr)
    sampled_count = int(sampled_count)

    # putting same rotations in a list one after another
    for rotation_offset in range(OFFSETS_PER_DATAPOINT):
        rotation_arr = []
        for position_index in range(sampled_count):
            datapoint_rotation = data_arr[position_index][rotation_offset]
            rotation_arr.append(datapoint_rotation)

        data_arr_same_rotations.append(rotation_arr)

    data_arr_tensor = torch.stack([torch.stack(pda) for pda in data_arr_same_rotations]).to(get_device())

    original_shape_data = data_arr_tensor.shape
    x_reshaped = data_arr_tensor.view(-1, original_shape_data[2])
    positional_encoding, thetas_encoding = autoencoder.encoder_training(x_reshaped)
    # reshape back
    thetas_encoding = thetas_encoding.view(original_shape_data[0], original_shape_data[1], -1)
    positional_encoding_reshaped = positional_encoding.view(original_shape_data[0], original_shape_data[1], -1)

    cdist_rotational_encoding_same_rotation = torch.cdist(thetas_encoding, thetas_encoding, p=2).mean()
    cdist_positional_encoding_same_rotation = torch.cdist(positional_encoding_reshaped, positional_encoding_reshaped,
                                                          p=2).mean()

    # average over each dimension of the embedding
    thetas_encoding = thetas_encoding.mean(dim=1)
    # duplicate to batch size
    thetas_encoding = thetas_encoding.to("cpu").detach().numpy()
    thetas_encoding = np.repeat(thetas_encoding[:, np.newaxis, :], sampled_count, axis=1)
    thetas_encoding = array_to_tensor(thetas_encoding).to(get_device())
    new_shape_data = thetas_encoding.shape

    # reshape again
    thetas_encoding = thetas_encoding.view(-1, new_shape_data[2])

    dec = autoencoder.decoder_training(positional_encoding, thetas_encoding)
    criterion = nn.MSELoss()
    loss_reconstruction_same_rotation = criterion(dec, x_reshaped)

    return loss_reconstruction_same_rotation, cdist_rotational_encoding_same_rotation, cdist_positional_encoding_same_rotation


def reconstruction_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2) -> any:
    global _cache_reconstruction_loss, _cache_thetas
    sampled_count = 200
    datapoints = storage.get_all_datapoints()
    sampled_count = min(sampled_count, len(datapoints))
    sampled_points = storage.sample_n_random_datapoints(sampled_count)

    loss_reconstruction_same_position = torch.tensor(0.0, device=get_device())
    cdist_positional_encoding_same_position = torch.tensor(0.0, device=get_device())

    data_arr = []
    for point in sampled_points:
        point_data_arr = []
        thetas_data_arr = []
        if point in _cache_reconstruction_loss:
            point_data_arr = _cache_reconstruction_loss[point]
        else:
            # creates list of inputs for that point
            for i in range(OFFSETS_PER_DATAPOINT):
                sampled_full_rotation = array_to_tensor(
                    storage.get_point_rotations_with_full_info_set_offset_concatenated(point, ROTATIONS_PER_FULL, i))
                point_data_arr.append(sampled_full_rotation)

            _cache_reconstruction_loss[point] = point_data_arr
            _cache_thetas[point] = thetas_data_arr

        data_arr.append(point_data_arr)

    loss_reconstruction_same_position, cdist_positional_encoding_same_position, cdist_rotational_encoding_same_position = same_position_handling(
        autoencoder, storage, data_arr)
    loss_reconstruction_same_rotation, cdist_rotational_encoding_same_rotation, cdist_positional_encoding_same_rotation = same_rotation_handling(
        autoencoder, storage, data_arr)

    return {
        "loss_reconstruction_same_position": loss_reconstruction_same_position,
        "cdist_positional_encoding_same_position": cdist_positional_encoding_same_position,
        "cdist_rotational_encoding_same_position": cdist_rotational_encoding_same_position,
        "loss_reconstruction_same_rotation": loss_reconstruction_same_rotation,
        "cdist_rotational_encoding_same_rotation": cdist_rotational_encoding_same_rotation,
        "cdist_positional_encoding_same_rotation": cdist_positional_encoding_same_rotation,
    }


def linearity_distance_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                                non_adjacent_sample_size: int, embedding_scaling_factor: float = 1) -> torch.Tensor:
    """
    Makes first degree connections be linearly distant from each other
    """
    # sampled_pairs = storage.sample_datapoints_adjacencies(non_adjacent_sample_size)
    non_adjacent_sample_size = min(non_adjacent_sample_size, len(storage.get_all_connections_only_datapoints()))
    sampled_pairs = storage.sample_adjacent_datapoints_connections(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []
    expected_distances = []

    for pair in sampled_pairs:
        # datapoint1 = storage.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        # datapoint2 = storage.get_datapoint_data_tensor_by_name_permuted(pair["end"])

        for j in range(4):
            datapoint1 = storage.get_datapoint_data_random_rotation_tensor_by_name(pair["start"])
            datapoint2 = storage.get_datapoint_data_random_rotation_tensor_by_name(pair["end"])
            expected_distance = pair["distance"]

            batch_datapoint1.append(datapoint1)
            batch_datapoint2.append(datapoint2)
            expected_distances.append(expected_distance)

    batch_datapoint1 = torch.stack(batch_datapoint1).to(get_device())
    batch_datapoint2 = torch.stack(batch_datapoint2).to(get_device())

    encoded_i, thetas_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j, thetas_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    predicted_distances = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    predicted_distances /= embedding_scaling_factor
    expected_distances = [distance for distance in expected_distances]
    expected_distances = torch.tensor(expected_distances, dtype=torch.float32).to(get_device())

    criterion = nn.MSELoss()

    non_adjacent_distance_loss = criterion(predicted_distances, expected_distances)
    return non_adjacent_distance_loss


def _train_autoencoder_abstraction_block(abstraction_block: AbstractionBlockSecondTrial, storage: StorageSuperset2,
                                         epochs: int,
                                         pretty_print: bool = False) -> AbstractionBlockSecondTrial:
    # PARAMETERS
    optimizer = optim.Adam(abstraction_block.parameters(), lr=0.00020, amsgrad=True)
    abstraction_block = abstraction_block.to(device=get_device())
    abstraction_block.train()

    num_epochs = epochs

    epoch_average_loss = 0
    scale_reconstruction_loss = 3

    non_adjacent_sample_size = 100

    average_loss_cdist_same_position = 0
    average_loss_reconstruction_same_position = 0
    average_loss_cdist_same_rotation = 0
    average_loss_reconstruction_same_rotation = 0
    average_loss_linearity = 0

    epoch_print_rate = 500

    if pretty_print:
        pretty_display_set(epoch_print_rate, "Epoch batch")
        pretty_display_start(0)

    SHUFFLE = 5
    for epoch in range(num_epochs):
        if epoch % SHUFFLE == 0:
            storage.build_permuted_data_random_rotations()

        epoch_loss = 0.0
        linearity_loss = torch.tensor(0.0, device=get_device())
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        losses_json = reconstruction_handling(
            abstraction_block,
            storage)
        # linearity_loss = linearity_distance_handling(abstraction_block, storage, non_adjacent_sample_size)

        reconstruction_loss_same_position = losses_json["loss_reconstruction_same_position"]
        cdist_positional_encoding_same_position = losses_json["cdist_positional_encoding_same_position"]
        cdist_positional_encoding_same_rotation = losses_json["cdist_positional_encoding_same_rotation"]
        reconstruction_loss_same_rotation = losses_json["loss_reconstruction_same_rotation"]
        cdist_rotational_encoding_same_rotation = losses_json["cdist_rotational_encoding_same_rotation"]
        cdist_rotational_encoding_same_position = losses_json["cdist_rotational_encoding_same_position"]

        ratio_position_encoding_loss = cdist_positional_encoding_same_position / cdist_positional_encoding_same_rotation
        ratio_rotation_encoding_loss = cdist_rotational_encoding_same_rotation / cdist_rotational_encoding_same_position

        reconstruction_loss_same_position *= scale_reconstruction_loss
        reconstruction_loss_same_rotation *= scale_reconstruction_loss

        accumulated_loss = reconstruction_loss_same_position + cdist_positional_encoding_same_position + reconstruction_loss_same_rotation + cdist_rotational_encoding_same_rotation + ratio_position_encoding_loss + ratio_rotation_encoding_loss
        accumulated_loss.backward()

        optimizer.step()

        reconstruction_loss_same_position /= scale_reconstruction_loss
        reconstruction_loss_same_rotation /= scale_reconstruction_loss

        epoch_average_loss += reconstruction_loss_same_position.item() + cdist_positional_encoding_same_position.item() + reconstruction_loss_same_rotation.item() + cdist_rotational_encoding_same_rotation.item() + linearity_loss.item()

        average_loss_reconstruction_same_position += reconstruction_loss_same_position.item()
        average_loss_cdist_same_position += cdist_positional_encoding_same_position.item()
        average_loss_reconstruction_same_rotation += reconstruction_loss_same_rotation.item()
        average_loss_cdist_same_rotation += cdist_rotational_encoding_same_rotation.item()
        average_loss_linearity += linearity_loss.item()

        if pretty_print:
            pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:

            epoch_average_loss /= epoch_print_rate

            average_loss_reconstruction_same_position /= epoch_print_rate
            average_loss_cdist_same_position /= epoch_print_rate
            average_loss_reconstruction_same_rotation /= epoch_print_rate
            average_loss_cdist_same_rotation /= epoch_print_rate
            average_loss_linearity /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs} ")
            print(
                f"RECONSTRUCTION LOSS POS:{average_loss_reconstruction_same_position} | CDIST LOSS POS:{average_loss_cdist_same_position} | RECONSTRUCTION LOSS ROT:{average_loss_reconstruction_same_rotation} | CDIST LOSS ROT:{average_loss_cdist_same_rotation} | LINEARITY LOSS:{average_loss_linearity}")
            print(f"RATIO POS:{1 / ratio_position_encoding_loss} | RATIO ROT:{1 / ratio_rotation_encoding_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            average_loss_reconstruction_same_position = 0
            average_loss_cdist_same_position = 0
            average_loss_reconstruction_same_rotation = 0
            average_loss_linearity = 0

            if pretty_print:
                pretty_display_reset()
                pretty_display_start(epoch)

    return abstraction_block


def train_abstraction_block_second_trial(abstraction_network: AbstractionBlockSecondTrial,
                                         storage: StorageSuperset2) -> AbstractionBlockSecondTrial:
    abstraction_network = _train_autoencoder_abstraction_block(abstraction_network, storage, 2501, True)
    return abstraction_network
