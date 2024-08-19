import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from triton.language import dtype

from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor, get_device
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display

import torch
import torch.nn as nn
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer


class AutoencoderExploration(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, embedding_size: int = 128, input_output_size: int = 512,
                 hidden_size: int = 512, num_blocks: int = 1):
        super(AutoencoderExploration, self).__init__()
        self.embedding_size = embedding_size

        self.input_layer = nn.Linear(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.manifold_encoder = _make_layer(hidden_size, embedding_size)
        self.manifold_decoder = _make_layer(embedding_size, hidden_size)

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


def non_adjacent_distance_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                                   non_adjacent_sample_size: int,
                                   scale_non_adjacent_distance_loss: float, distance_per_neuron: float) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other
    """
    sampled_pairs = storage.sample_datapoints_adjacencies(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []

    for pair in sampled_pairs:
        datapoint1 = storage.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        datapoint2 = storage.get_datapoint_data_tensor_by_name_permuted(pair["end"])

        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1).to(get_device())
    batch_datapoint2 = torch.stack(batch_datapoint2).to(get_device())

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] * distance_per_neuron * embedding_size for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance, dtype=torch.float32).to(get_device())

    criterion = nn.MSELoss()

    non_adjacent_distance_loss = criterion(distance, expected_distance) * scale_non_adjacent_distance_loss
    return non_adjacent_distance_loss


def train_autoencoder_with_distance_constraint(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                                               epochs: int) -> BaseAutoencoderModel:
    # PARAMETERS
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005)

    num_epochs = epochs

    scale_reconstruction_loss = 1
    scale_adjacent_distance_loss = 0.5
    scale_non_adjacent_distance_loss = 0.5

    non_adjacent_sample_size = 1000

    epoch_average_loss = 0
    reconstruction_average_loss = 0
    non_adjacent_average_loss = 0

    epoch_print_rate = 100
    DISTANCE_CONSTANT_PER_NEURON = 0.005

    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(get_device())
    autoencoder = autoencoder.to(get_device())

    set_pretty_display(epoch_print_rate, "Epoch batch")
    pretty_display_start(0)

    SHUFFLE_RATE = 5
    for epoch in range(num_epochs):

        reconstruction_loss = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, scale_reconstruction_loss)
        reconstruction_loss.backward()

        # NON-ADJACENT DISTANCE LOSS
        non_adjacent_distance_loss = torch.tensor(0.0)
        non_adjacent_distance_loss = non_adjacent_distance_handling(autoencoder, storage, non_adjacent_sample_size,
                                                                    scale_non_adjacent_distance_loss,
                                                                    distance_per_neuron=DISTANCE_CONSTANT_PER_NEURON)
        non_adjacent_distance_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + non_adjacent_distance_loss.item()

        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()

        pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate
            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs}")
            # print(f"average distance between adjacent: {average_distance_adjacent}")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            non_adjacent_average_loss = 0

            pretty_display_reset()
            pretty_display_start(epoch)

    return autoencoder


def run_autoencoder_network(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2):
    storage.build_permuted_data_random_rotations_rotation0()
    autoencoder = train_autoencoder_with_distance_constraint(autoencoder, storage, epochs=3001)

    return autoencoder


def generate_autoencoder_ai():
    return AutoencoderExploration().to(get_device())
