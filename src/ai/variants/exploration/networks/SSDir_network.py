import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData
from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.params import MANIFOLD_SIZE, DIRECTION_THETAS
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, pretty_display_set, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import direction_thetas_to_radians, \
    angle_percent_to_thetas_normalized_cached, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm


class SSDirNetwork(nn.Module):
    def __init__(self, input_size=MANIFOLD_SIZE, hidden_size=512, output_size=DIRECTION_THETAS, dropout_rate=0.3,
                 num_blocks=1):
        super(SSDirNetwork, self).__init__()

        self.input_layer = nn.Linear(input_size * 2, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)

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


datapoint_embeddings_cache = {}


def direction_loss(direction_network, storage: StorageSuperset2, sample_rate):
    loss = torch.tensor(0.0)

    datapoints: List[str] = storage.sample_n_random_datapoints(sample_rate)
    start_embeddings_batch = []
    end_embeddings_batch = []

    target_thetas_batch = []

    for datapoint in datapoints:
        if datapoint in datapoint_embeddings_cache:
            start_data = datapoint_embeddings_cache[datapoint]["start_data"]
            end_data = datapoint_embeddings_cache[datapoint]["end_data"]
            thetas_target = datapoint_embeddings_cache[datapoint]["thetas_target"]
            start_embeddings_batch.extend(start_data)
            end_embeddings_batch.extend(end_data)
            target_thetas_batch.extend(thetas_target)
            continue

        start_data_arr = []
        end_data_arr = []
        thetas_target_arr = []

        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections_authentic(datapoint)
        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]
            direction = connections_to_point[j]["direction"]
            if direction == None:
                print("None direction found")
            direction = torch.tensor(direction, dtype=torch.float32)

            final_radian = coordinate_pair_to_radians_cursed_tranform(direction[0], direction[1])
            radian_percent = radians_to_percent(final_radian)
            thetas_target = angle_percent_to_thetas_normalized_cached(radian_percent, DIRECTION_THETAS)

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            thetas_target_arr.append(thetas_target)

        start_embeddings_batch.extend(start_data_arr)
        end_embeddings_batch.extend(end_data_arr)
        target_thetas_batch.extend(thetas_target_arr)

        datapoint_embeddings_cache[datapoint] = {
            "start_data": start_data_arr,
            "end_data": end_data_arr,
            "thetas_target": thetas_target_arr
        }

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(get_device())
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(get_device())

    target_thetas_batch = torch.stack(target_thetas_batch).to(get_device())

    # Log softmax output
    output_thetas_batch = direction_network.forward_training(start_embeddings_batch, end_embeddings_batch)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    loss = criterion(output_thetas_batch, target_thetas_batch)
    return loss


def _train_SSDir_network(direction_network, storage: StorageSuperset2, num_epochs: int):
    optimizer = optim.Adam(direction_network.parameters(), lr=0.0005, amsgrad=True)

    scale_direction_loss = 1

    epoch_average_loss = 0
    sample_rate = 200

    epoch_print_rate = 250
    storage.build_permuted_data_random_rotations_rotation0()

    pretty_display_set(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    SHUFFLE = 2
    for epoch in range(num_epochs):
        if epoch % SHUFFLE == 0:
            storage.build_permuted_data_random_rotations()

        pretty_display(epoch % epoch_print_rate)

        epoch_loss = 0.0

        optimizer.zero_grad()

        loss = direction_loss(direction_network, storage, sample_rate)
        # loss = direction_loss_v2(direction_network, storage, sample_rate)
        loss = loss * scale_direction_loss
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


def datapoints_to_manifold(datapoints, autoencoder: BaseAutoencoderModel):
    autoencoder.eval()
    autoencoder = autoencoder.to(get_device())

    for datapoint in datapoints:
        data = array_to_tensor(datapoint["data"]).to(get_device())
        datapoint["data"] = autoencoder.encoder_inference(data).tolist()

    return datapoints


def train_SSDirection(SSDir_network: SSDirNetwork, storage: StorageSuperset2):
    SSDir_network = SSDir_network.to(get_device())
    direction_network = _train_SSDir_network(SSDir_network, storage, num_epochs=5000)
    return direction_network
