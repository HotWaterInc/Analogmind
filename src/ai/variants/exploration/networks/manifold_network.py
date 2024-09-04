import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.params import MANIFOLD_SIZE, THRESHOLD_MANIFOLD_PERMUTATION_LOSS, \
    THRESHOLD_MANIFOLD_NON_ADJACENT_LOSS
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor, get_device
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, pretty_display_set
import torch
import torch.nn as nn
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer_no_batchnorm_leaky


class ManifoldNetwork(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, embedding_size: int = MANIFOLD_SIZE, input_output_size: int = 512,
                 hidden_size: int = 2048, num_blocks: int = 2):
        super(ManifoldNetwork, self).__init__()
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


def adjacent_distance_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                               adjacent_sample_size: int) -> tuple[torch.Tensor, float]:
    """
    Keeps adjacent pairs close to each other
    """
    sampled_pairs = storage.sample_adjacent_datapoints_connections(adjacent_sample_size)

    adjacent_distance_loss = torch.tensor(0.0, device=get_device())
    average_distance = 0
    batch_datapoint1 = []
    batch_datapoint2 = []
    batch_distance = []

    for pair in sampled_pairs:
        # keep adjacent close to each other
        data_point1 = storage.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        data_point2 = storage.get_datapoint_data_tensor_by_name_permuted(pair["end"])
        distance = pair["distance"]

        batch_datapoint1.append(data_point1)
        batch_datapoint2.append(data_point2)
        batch_distance.append(distance)

    batch_datapoint1 = torch.stack(batch_datapoint1).to(get_device())
    batch_datapoint2 = torch.stack(batch_datapoint2).to(get_device())

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    distances_embeddings = torch.norm((encoded_i - encoded_j), p=2)
    batch_distance = torch.tensor(batch_distance, dtype=torch.float32).to(get_device())
    adjusted_distances = torch.div(distances_embeddings, batch_distance)

    average_distance = adjusted_distances.mean().item()
    adjacent_distance_loss = distances_embeddings.mean()

    return adjacent_distance_loss, average_distance


def permutation_adjustion_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                                   samples: int) -> torch.Tensor:
    """
    Keeps the permutation of the data points close to each other
    """
    datapoint: List[str] = storage.sample_n_random_datapoints(samples)
    datapoints_data = [storage.get_datapoint_data_tensor_by_name(name) for name in datapoint]
    accumulated_loss = torch.tensor(0.0, device=get_device())

    datapoints_data = torch.stack(datapoints_data).to(get_device())

    original_shape = datapoints_data.shape
    to_encode = datapoints_data.view(-1, original_shape[-1])
    encoded = autoencoder.encoder_training(to_encode)
    encoded = encoded.view(original_shape[0], original_shape[1], encoded.shape[-1])
    accumulated_loss = torch.cdist(encoded, encoded, p=2).mean()

    return accumulated_loss


def non_adjacent_distance_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                                   non_adjacent_sample_size: int,
                                   distance_scaling_factor: float, embedding_scaling_factor: float = 1) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other

    distance scaling factors accounts for the range in which MSE is calculated, helps to avoid exploding or vanishing losses

    embedding scaling factor scales the embeddings to be further apart or closer together, without actually affecting the MSE loss
        * If we want generally smaller embeddings without leading to MSE collapsing to 0, we can use this parameter

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

    distances_embeddings = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    # making the distance of the embeddings smaller for example, forces them to become bigger in order to match the normal distance
    distances_embeddings = distances_embeddings / embedding_scaling_factor

    expected_distance = [pair["distance"] * distance_scaling_factor for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance, dtype=torch.float32).to(get_device())

    # Distance scaling factor controls how far away the embeddings actually are
    # For big and small distances, it readjusts MSE to not explode or vanish
    criterion = nn.MSELoss()
    non_adjacent_distance_loss = criterion(distances_embeddings, expected_distance)

    return non_adjacent_distance_loss


def _train_autoencoder_with_distance_constraint(manifold_network: BaseAutoencoderModel, storage: StorageSuperset2,
                                                epochs: int, stop_at_threshold: bool = False) -> BaseAutoencoderModel:
    # PARAMETERS
    optimizer = optim.Adam(manifold_network.parameters(), lr=0.00020)

    num_epochs = epochs

    scale_reconstruction_loss = 1
    scale_non_adjacent_distance_loss = 10
    scale_adjacent_distance_loss = 1

    non_adjacent_sample_size = 1000
    adjacent_sample_size = 100
    permutation_sample_size = 100

    epoch_average_loss = 0

    reconstruction_average_loss = 0
    non_adjacent_average_loss = 0
    adjacent_average_loss = 0
    permutation_average_loss = 0

    epoch_print_rate = 1000
    DISTANCE_SCALING_FACTOR = 1
    EMBEDDING_SCALING_FACTOR = 0.1

    storage.build_permuted_data_random_rotations_rotation0()
    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(get_device())
    manifold_network = manifold_network.to(get_device())

    print("STARTED TRAINING")
    pretty_display_reset()
    pretty_display_set(epoch_print_rate, "Epoch batch")
    pretty_display_start(0)

    if stop_at_threshold:
        num_epochs = int(1e7)

    SHUFFLE_RATE = 2
    for epoch in range(num_epochs):
        if epoch % SHUFFLE_RATE == 0:
            storage.build_permuted_data_random_rotations()
            # storage.build_permuted_data_random_rotations_rotation0()

        reconstruction_loss = torch.tensor(0.0)
        adjacent_distance_loss = torch.tensor(0.0)
        non_adjacent_distance_loss = torch.tensor(0.0)
        permutation_adjustion_loss = torch.tensor(0.0)

        epoch_loss = 0.0
        optimizer.zero_grad()
        accumulated_loss = torch.tensor(0.0, device=get_device())

        # RECONSTRUCTION LOSS
        # reconstruction_loss = reconstruction_handling(autoencoder, train_data, scale_reconstruction_loss)

        # ADJACENT DISTANCE LOSS
        # adjacent_distance_loss, average_distance_adjacent = adjacent_distance_handling(autoencoder, storage,
        #                                                                                adjacent_sample_size)
        # NON-ADJACENT DISTANCE LOSS
        non_adjacent_distance_loss = non_adjacent_distance_handling(manifold_network, storage, non_adjacent_sample_size,
                                                                    distance_scaling_factor=DISTANCE_SCALING_FACTOR,
                                                                    embedding_scaling_factor=EMBEDDING_SCALING_FACTOR)
        # PERMUTATION ADJUST LOSS
        permutation_adjustion_loss = permutation_adjustion_handling(manifold_network, storage, permutation_sample_size)

        accumulated_loss = reconstruction_loss + non_adjacent_distance_loss + adjacent_distance_loss + permutation_adjustion_loss
        accumulated_loss.backward()
        optimizer.step()

        epoch_loss += reconstruction_loss.item() + non_adjacent_distance_loss.item() + adjacent_distance_loss.item() + \
                      permutation_adjustion_loss.item()

        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()
        adjacent_average_loss += adjacent_distance_loss.item()
        permutation_average_loss += permutation_adjustion_loss.item()

        pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate

            reconstruction_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate
            adjacent_average_loss /= epoch_print_rate
            permutation_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs}")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | PERMUTATION LOSS:{permutation_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            if non_adjacent_average_loss < THRESHOLD_MANIFOLD_NON_ADJACENT_LOSS and permutation_average_loss < THRESHOLD_MANIFOLD_PERMUTATION_LOSS and stop_at_threshold:
                print(f"Stopping at epoch {epoch} with loss {epoch_average_loss} because of threshold")
                break

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            non_adjacent_average_loss = 0
            adjacent_average_loss = 0
            permutation_average_loss = 0

            pretty_display_reset()
            pretty_display_start(epoch)

    return manifold_network


def train_manifold_network_until_thresholds(manifold_network: BaseAutoencoderModel, storage: StorageSuperset2):
    manifold_network = _train_autoencoder_with_distance_constraint(
        manifold_network=manifold_network,
        storage=storage,
        epochs=-1,
        stop_at_threshold=True
    )

    return manifold_network


def train_manifold_network(manifold_network: BaseAutoencoderModel, storage: StorageSuperset2):
    manifold_network = _train_autoencoder_with_distance_constraint(
        manifold_network=manifold_network,
        storage=storage,
        epochs=15000,
        stop_at_threshold=False
    )

    return manifold_network
