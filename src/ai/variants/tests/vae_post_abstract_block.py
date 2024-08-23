import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, pretty_display_set

import torch
import torch.nn as nn
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer_no_batchnorm_leaky


class VAEOverAbstraction(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, embedding_size: int = 128, input_output_size: int = 96,
                 hidden_size: int = 1024, num_blocks: int = 1):
        super(VAEOverAbstraction, self).__init__()
        self.embedding_size = embedding_size

        self.input_layer = nn.Linear(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.manifold_encoder = _make_layer_no_batchnorm_leaky(hidden_size, embedding_size)
        self.manifold_decoder = _make_layer_no_batchnorm_leaky(embedding_size, hidden_size)

        self.decoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.output_layer = nn.Linear(hidden_size, input_output_size)
        self.leaky_relu = nn.LeakyReLU()

    def encoder_training(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.encoding_blocks:
            x = block(x)
        x = self.manifold_encoder(x)

        return x

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_training(x)

    def decoder_training(self, x: torch.Tensor) -> torch.Tensor:
        x = self.manifold_decoder(x)
        for block in self.decoding_blocks:
            x = block(x)
        x = self.output_layer(x)

        return x

    def decoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(x)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_training(x)
        decoded = self.decoder_training(encoded)
        return decoded

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)

    def get_embedding_size(self) -> int:
        return self.embedding_size


def reconstruction_handling(autoencoder: BaseAutoencoderModel, data: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    dec = autoencoder.forward_training(data)
    criterion = nn.MSELoss()
    return criterion(dec, data) * scale_reconstruction_loss


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

    batch_datapoint1 = torch.stack(batch_datapoint1).to(device)
    batch_datapoint2 = torch.stack(batch_datapoint2).to(device)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    # puts distance on cpu
    distance = torch.sum(torch.norm((encoded_i - encoded_j), p=2)).to("cpu")

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

    batch_datapoint1 = torch.stack(batch_datapoint1).to(device)
    batch_datapoint2 = torch.stack(batch_datapoint2).to(device)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] * distance_per_neuron * embedding_size for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance).to(device)

    criterion = nn.MSELoss()

    non_adjacent_distance_loss = criterion(distance, expected_distance) * scale_non_adjacent_distance_loss
    return non_adjacent_distance_loss


def train_autoencoder_with_distance_constraint(autoencoder: BaseAutoencoderModel, epochs: int) -> BaseAutoencoderModel:
    global storage_raw
    # PARAMETERS
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.00025, amsgrad=True)

    num_epochs = epochs
    autoencoder = autoencoder.to(device)

    scale_reconstruction_loss = 1
    scale_adjacent_distance_loss = 0.5
    scale_non_adjacent_distance_loss = 0.5

    adjacent_sample_size = 25
    non_adjacent_sample_size = 300

    epoch_average_loss = 0

    reconstruction_average_loss = 0
    adjacent_average_loss = 0
    non_adjacent_average_loss = 0

    epoch_print_rate = 1000
    DISTANCE_CONSTANT_PER_NEURON = 0.005
    SHUFFLE_RATE = 10

    storage.build_permuted_data_random_rotations_rotation0()
    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(device)
    autoencoder = autoencoder.to(device)

    pretty_display_set(epoch_print_rate, "Epoch batch")
    pretty_display_start(0)

    for epoch in range(num_epochs):
        if (epoch % SHUFFLE_RATE == 0):
            storage.build_permuted_data_random_rotations()
            # storage.build_permuted_data_random_rotations_rotation0()

            train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))
            train_data = train_data.to(device)

        reconstruction_loss = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, scale_reconstruction_loss)
        reconstruction_loss.backward()

        # ADJACENT DISTANCE LOSS
        adjacent_distance_loss = torch.tensor(0.0)
        adjacent_distance_loss, average_distance_adjacent = adjacent_distance_handling(autoencoder,
                                                                                       adjacent_sample_size,
                                                                                       scale_adjacent_distance_loss)
        adjacent_distance_loss.backward()

        # NON-ADJACENT DISTANCE LOSS
        non_adjacent_distance_loss = torch.tensor(0.0)
        non_adjacent_distance_loss = non_adjacent_distance_handling(autoencoder, non_adjacent_sample_size,
                                                                    scale_non_adjacent_distance_loss,
                                                                    distance_per_neuron=DISTANCE_CONSTANT_PER_NEURON)
        non_adjacent_distance_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + adjacent_distance_loss.item() + non_adjacent_distance_loss.item()

        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        adjacent_average_loss += adjacent_distance_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()

        pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            adjacent_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate

            print("")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            adjacent_average_loss = 0
            non_adjacent_average_loss = 0

            pretty_display_reset()
            pretty_display_start(epoch)

    return autoencoder


def run_tests(autoencoder):
    global storage

    evaluate_reconstruction_error_super(autoencoder, storage, rotations0=False)
    avg_distance_adj = evaluate_distances_between_pairs_super(autoencoder, storage, rotations0=False)
    evaluate_adjacency_properties_super(autoencoder, storage, avg_distance_adj, rotation0=False)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("autoenc_dynamic10k.pth")
    autoencoder = load_manually_saved_ai("vae_post_abstraction_block_saved.pth")
    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = VAEOverAbstraction()
    train_autoencoder_with_distance_constraint(autoencoder, epochs=15001)
    save_ai_manually("vae_post_abstraction_block", autoencoder)
    run_tests(autoencoder)


def run_autoencoder_post_abstract_block_img1() -> None:
    global storage
    global permutor
    permutor = load_manually_saved_ai("abstract_block_img1_saved.pth")
    permutor.eval()
    permutor = permutor.to(device)

    grid_data = 5

    storage.load_raw_data_from_others(f"data{grid_data}x{grid_data}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_data}x{grid_data}_connections.json")

    storage.set_permutor(permutor)
    storage.build_permuted_data_raw_abstraction_block_1img()

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = StorageSuperset2()
permutor = None

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
