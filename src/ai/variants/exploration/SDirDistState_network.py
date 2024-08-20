import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData, \
    distance_percent_to_distance_thetas
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually, load_custom_ai, \
    load_other_ai
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, set_pretty_display, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import thetas_to_radians, \
    angle_percent_to_thetas_normalized_cached, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan, degrees_to_percent, normalize_direction
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm

DIRECTION_THETAS_SIZE = 36
DISTANCE_THETAS_SIZE = 100
MANIFOLD_SIZE = 128
MAX_DISTANCE = 3


class SDirDistState(nn.Module):
    def __init__(self, manifold_size=MANIFOLD_SIZE, direction_thetas_size=DIRECTION_THETAS_SIZE,
                 distance_thetas_size=DISTANCE_THETAS_SIZE,
                 hidden_size=512,
                 dropout_rate=0.3,
                 num_blocks=1):
        super(SDirDistState, self).__init__()

        self.input_layer = nn.Linear(manifold_size + direction_thetas_size + distance_thetas_size, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, manifold_size)

    def _forward_pass(self, x, y, z):
        inpt = torch.cat((x, y, z), dim=1)

        out = self.input_layer(inpt)
        for block in self.blocks:
            out = block(out)

        output = self.output_layer(out)
        return output

    def forward_training(self, x, y, z):
        output = self._forward_pass(x, y, z)
        return output

    def forward(self, x, y, z):
        output = self._forward_pass(x, y, z)
        return output


datapoint_embeddings_cache = {}


def SDirDistState_loss(direction_network, storage: StorageSuperset2, sample_rate):
    loss = torch.tensor(0.0)

    datapoints: List[str] = storage.sample_n_random_datapoints(sample_rate)

    start_embeddings_batch = []
    direction_thetas_batch = []
    distance_thetas_batch = []
    target_embeddings_batch = []

    for datapoint in datapoints:
        if datapoint in datapoint_embeddings_cache:
            start_data = datapoint_embeddings_cache[datapoint]["start_data"]
            end_data = datapoint_embeddings_cache[datapoint]["end_data"]
            direction_thetas = datapoint_embeddings_cache[datapoint]["direction_thetas"]
            distance_thetas = datapoint_embeddings_cache[datapoint]["distance_thetas"]

            start_embeddings_batch.extend(start_data)
            direction_thetas_batch.extend(direction_thetas)
            distance_thetas_batch.extend(distance_thetas)
            target_embeddings_batch.extend(end_data)
            continue

        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
        start_data_arr = []
        end_data_arr = []
        direction_thetas_arr = []
        distance_thetas_arr = []
        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]
            direction = connections_to_point[j]["direction"]
            direction = torch.tensor(direction, dtype=torch.float32)
            direction_normalized = direction / torch.norm(direction, p=2, dim=0, keepdim=True)
            distance = connections_to_point[j]["distance"]
            distance_percent = distance / MAX_DISTANCE

            direction_angle = direction_to_degrees_atan(direction_normalized)
            direction_percent = degrees_to_percent(direction_angle)

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
            direction_theta_form = angle_percent_to_thetas_normalized_cached(direction_percent, DIRECTION_THETAS_SIZE)
            distance_theta_form = distance_percent_to_distance_thetas(distance_percent, DISTANCE_THETAS_SIZE)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            direction_thetas_arr.append(direction_theta_form)
            distance_thetas_arr.append(distance_theta_form)

        start_embeddings_batch.extend(start_data_arr)
        direction_thetas_batch.extend(direction_thetas_arr)
        distance_thetas_batch.extend(distance_thetas_arr)
        target_embeddings_batch.extend(end_data_arr)

        datapoint_embeddings_cache[datapoint] = {
            "start_data": start_data_arr,
            "end_data": end_data_arr,
            "direction_thetas": direction_thetas_arr,
            "distance_thetas": distance_thetas_arr
        }

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(get_device())
    direction_thetas_batch = torch.stack(direction_thetas_batch).to(get_device())
    distance_thetas_batch = torch.stack(distance_thetas_batch).to(get_device())
    target_embeddings_batch = torch.stack(target_embeddings_batch).to(get_device())

    predicted_embeddings_batch = direction_network.forward_training(start_embeddings_batch, direction_thetas_batch,
                                                                    distance_thetas_batch)

    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()

    loss = criterion(predicted_embeddings_batch, target_embeddings_batch)
    loss += criterion2(predicted_embeddings_batch, target_embeddings_batch)

    return loss


def train_SDirDistState(SDDS: SDirDistState, storage: StorageSuperset2, num_epochs):
    optimizer = optim.Adam(SDDS.parameters(), lr=0.001, amsgrad=True)

    scale_direction_loss = 10

    epoch_average_loss = 0
    sample_rate = 500

    epoch_print_rate = 100

    storage.build_permuted_data_random_rotations_rotation0()
    set_pretty_display(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    for epoch in range(num_epochs):
        pretty_display(epoch % epoch_print_rate)

        epoch_loss = 0.0

        optimizer.zero_grad()

        loss = SDirDistState_loss(SDDS, storage, sample_rate) * scale_direction_loss
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() / scale_direction_loss
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            print("")
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_average_loss:.4f}')
            epoch_average_loss = 0  # Reset for the next average calculation

            pretty_display_reset()
            pretty_display_start(epoch)

    print("")
    return SDDS


def storage_to_manifold(storage: StorageSuperset2, autoencoder: BaseAutoencoderModel):
    autoencoder.eval()
    autoencoder = autoencoder.to(get_device())

    storage.set_permutor(autoencoder)
    storage.build_permuted_data_raw_abstraction_autoencoder_manifold()


def run_SDirDistState(SDisDistState_network: SDirDistState,
                      storage: StorageSuperset2):
    direction_network = train_SDirDistState(SDisDistState_network, storage, num_epochs=2000)
    return direction_network
