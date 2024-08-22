import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData, calculate_coords_distance
from src.ai.variants.exploration.utils import DISTANCE_THETAS_SIZE, MAX_DISTANCE
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, set_pretty_display, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import thetas_to_radians, \
    angle_percent_to_thetas_normalized_cached, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan, distance_percent_to_distance_thetas, distance_thetas_to_distance_percent
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm


class NeighborhoodNetworkThetasFull(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, output_size=DISTANCE_THETAS_SIZE, dropout_rate=0.3,
                 num_blocks=1):
        super(NeighborhoodNetworkThetasFull, self).__init__()

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
        return output

    def forward_training(self, x, y):
        output = self._forward_pass(x, y)
        log_softmax = F.log_softmax(output, dim=1)
        return log_softmax

    def forward(self, x, y):
        output = self._forward_pass(x, y)
        output = F.softmax(output, dim=1)
        return output


def distance_loss(neighborgood_network, storage, sample_rate):
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
            if real_life_distance > MAX_DISTANCE:
                real_life_distance = MAX_DISTANCE - 0.01

            thetas_target = distance_percent_to_distance_thetas(real_life_distance / MAX_DISTANCE,
                                                                DISTANCE_THETAS_SIZE)
            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)

            start_embeddings_batch.append(start_data)
            end_embeddings_batch.append(end_data)
            target_distances_batch.append(thetas_target)

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(get_device())
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(get_device())
    target_distances_batch = torch.stack(target_distances_batch).to(get_device())

    predicted_distances = neighborgood_network.forward_training(start_embeddings_batch, end_embeddings_batch)

    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    loss = criterion(predicted_distances, target_distances_batch)
    return loss


def _train_neighborhood_network(direction_network, storage, num_epochs,
                                pretty_print=True) -> NeighborhoodNetworkThetasFull:
    optimizer = optim.Adam(direction_network.parameters(), lr=0.0005, amsgrad=True)
    # increase size or LR

    scale_direction_loss = 1
    epoch_average_loss = 0
    epoch_print_rate = 500

    storage.build_permuted_data_random_rotations_rotation0()
    set_pretty_display(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    SHUFFLE = 5

    for epoch in range(num_epochs):
        if epoch % SHUFFLE == 0:
            storage.build_permuted_data_random_rotations_rotation0()

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


def run_neighborhood_network_thetas_full(neighborhood_network: NeighborhoodNetworkThetasFull,
                                         storage: StorageSuperset2) -> NeighborhoodNetworkThetasFull:
    storage.build_permuted_data_random_rotations_rotation0()
    neighborhood_network = _train_neighborhood_network(neighborhood_network, storage, 1500, True)
    return neighborhood_network
