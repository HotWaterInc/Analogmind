import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData, calculate_coords_distance
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, set_pretty_display, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import angle_to_thetas, thetas_to_radians, \
    angle_percent_to_thetas_normalized, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm


class NeighborhoodNetwork(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, output_size=1, dropout_rate=0.3, num_blocks=1):
        super(NeighborhoodNetwork, self).__init__()

        self.input_layer = nn.Linear(input_size * 2, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def _forward_pass(self, x, y):
        inpt = torch.cat((x, y), dim=1)

        out = self.input_layer(inpt)
        for block in self.blocks:
            out = block(out)

        output = self.output_layer(out)
        output = self.sigmoid(output) * 3  # clamping between 0 and 3
        # print(output)
        return output

    def forward_training(self, x, y):
        output = self._forward_pass(x, y)
        return output

    def forward(self, x, y):
        output = self._forward_pass(x, y)
        return output


def embedding_policy(data):
    global autoencoder
    start_embedding = data
    # start_embedding = autoencoder.encoder_inference(data)
    return start_embedding


def distance_loss(direction_network, storage, sample_rate):
    loss = torch.tensor(0.0)

    raw_env = storage.get_raw_environment_data()
    if sample_rate == None:
        sample_rate = len(raw_env)
    sample_rate = min(sample_rate, len(raw_env))

    datapoints: List[str] = storage.sample_n_random_datapoints(sample_rate)
    start_embeddings_batch = []
    end_embeddings_batch = []

    target_distances_batch = []

    for datapoint in datapoints:
        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]

            start_coords = storage.get_datapoint_metadata_coords(start)
            end_coords = storage.get_datapoint_metadata_coords(end)
            real_life_distance = calculate_coords_distance(start_coords, end_coords)

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)
            start_embedding = embedding_policy(start_data)
            end_embedding = embedding_policy(end_data)

            start_embeddings_batch.append(start_embedding)
            end_embeddings_batch.append(end_embedding)
            target_distances_batch.append(real_life_distance)

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(get_device())
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(get_device())
    target_distances_batch = torch.tensor(target_distances_batch).to(get_device())

    predicted_distances = direction_network(start_embeddings_batch, end_embeddings_batch).squeeze(1)

    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    loss = torch.tensor(0.0, device=get_device())

    # loss += criterion(predicted_distances, target_distances_batch)
    loss += criterion2(predicted_distances, target_distances_batch)

    return loss


def _train_neighborhood_network(direction_network, storage, num_epochs, pretty_print=True) -> NeighborhoodNetwork:
    optimizer = optim.Adam(direction_network.parameters(), lr=0.0005, amsgrad=True)

    scale_direction_loss = 1
    epoch_average_loss = 0
    epoch_print_rate = 100

    storage.build_permuted_data_random_rotations_rotation0()
    set_pretty_display(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    for epoch in range(num_epochs):
        pretty_display(epoch % epoch_print_rate)
        epoch_loss = 0.0
        optimizer.zero_grad()

        loss = distance_loss(direction_network, storage, sample_rate=None) * scale_direction_loss
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            print("")
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_average_loss:.4f}')
            epoch_average_loss = 0  # Reset for the next average calculation

            pretty_display_reset()
            pretty_display_start(epoch)

    return direction_network


def generate_new_ai() -> NeighborhoodNetwork:
    return NeighborhoodNetwork()


def run_neighborhood_network(neighborhood_network: NeighborhoodNetwork,
                             storage: StorageSuperset2) -> NeighborhoodNetwork:
    storage.build_permuted_data_random_rotations_rotation0()
    neighborhood_network = _train_neighborhood_network(neighborhood_network, storage, 2500, True)
    return neighborhood_network


def load_storage_data(storage: StorageSuperset2) -> StorageSuperset2:
    dataset_grid = 5
    storage.load_raw_data_from_others(f"data{dataset_grid}x{dataset_grid}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{dataset_grid}x{dataset_grid}_connections.json")
    # selects first rotation
    storage.build_permuted_data_random_rotations_rotation0()
    return storage
