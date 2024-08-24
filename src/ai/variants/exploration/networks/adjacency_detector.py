import math
from random import sample

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from triton.language import dtype
import random
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData, calculate_coords_distance
from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.params import THRESHOLD_ADJACENCY_DETECTOR, ROTATIONS
from src.ai.variants.exploration.utils_pure_functions import sample_n_elements
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, pretty_display_set, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import direction_thetas_to_radians, \
    angle_percent_to_thetas_normalized_cached, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer


class AdjacencyDetector(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=2, dropout_rate=0.6, num_blocks=1):
        super(AdjacencyDetector, self).__init__()
        self.input_layer = _make_layer(input_size * 2, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def _forward_pass(self, x, y):
        inpt = torch.cat((x, y), dim=1)

        out = self.input_layer(inpt)
        for block in self.blocks:
            out = block(out)

        output = self.output_layer(out)
        output = F.softmax(output, dim=1)

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


def adjacency_detector_loss(direction_network, storage, sample_rate=None):
    connections: RawConnectionData = storage.get_all_connections_only_datapoints_authenticity_filter(
        authentic_distance=True)
    if sample_rate == None:
        sample_rate = len(connections)
    sample_rate = min(sample_rate, len(connections))

    connections: List[RawConnectionData] = sample_n_elements(connections, sample_rate)

    start_embeddings_batch = []
    end_embeddings_batch = []
    target_probabilities_batch = []

    for connection in connections:
        start = connection["start"]
        end = connection["end"]
        distance = connection["distance"]

        start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
        end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)

        if distance > 0.9:
            # false
            target_probabilities_batch.append([0, 1])
            start_embeddings_batch.append(start_data)
            end_embeddings_batch.append(end_data)
        elif distance < 0.4:
            # true
            # Doubling explained in the next comment
            target_probabilities_batch.append([1, 0])
            target_probabilities_batch.append([1, 0])
            start_embeddings_batch.append(start_data)
            end_embeddings_batch.append(end_data)
            start_embeddings_batch.append(start_data)
            end_embeddings_batch.append(end_data)
        else:
            continue

    # select random sets of datapoints as counter example of far away non adjacent points
    # There is always the possiblity of selecting 2 datapoints which are actually adjacent, but the probability is really low
    # We can double the number of true cases to compensate for the eventual false cases we select here which are actually adjacecnt
    datapoints: List[str] = storage.get_all_datapoints()
    sample_size = 300
    sample_size = min(sample_size, len(datapoints))
    indices = torch.randperm(len(datapoints))[:sample_size]
    pairs = [(datapoints[i], datapoints[j]) for i, j in indices.view(int(sample_size / 2), 2).tolist()]

    for pair in pairs:
        start = pair[0]
        end = pair[1]
        start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
        end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)
        start_embedding = embedding_policy(start_data)
        end_embedding = embedding_policy(end_data)

        start_embeddings_batch.append(start_embedding)
        end_embeddings_batch.append(end_embedding)
        # false case
        target_probabilities_batch.append([0, 1])

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(get_device())
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(get_device())
    target_probabilities_batch = torch.tensor(target_probabilities_batch, dtype=torch.float32).to(get_device())
    predicted_adjacencies = direction_network(start_embeddings_batch, end_embeddings_batch)

    criterion = nn.BCELoss()
    loss = criterion(predicted_adjacencies, target_probabilities_batch)
    return loss


def _train_adjacency_network(adjacency_network, storage, num_epochs,
                             pretty_print=True, stop_at_threshold: bool = False) -> AdjacencyDetector:
    optimizer = optim.Adam(adjacency_network.parameters(), lr=0.0006, amsgrad=True)

    epoch_average_loss = 0
    epoch_print_rate = 100

    storage.build_permuted_data_random_rotations_rotation0()

    pretty_display_set(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    if stop_at_threshold:
        num_epochs = 1e7

    for epoch in range(num_epochs):
        pretty_display(epoch % epoch_print_rate)

        loss = torch.tensor(0.0, device=get_device())
        epoch_loss = 0.0

        optimizer.zero_grad()

        ITERATIONS = 1
        for i in range(ITERATIONS):
            rand_dir = random.randint(0, ROTATIONS - 1)
            storage.build_permuted_data_random_rotations_rotation_N_with_noise(rand_dir)

            loss += adjacency_detector_loss(adjacency_network, storage)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / ITERATIONS
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            print("")
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_average_loss:.4f}')

            if epoch_average_loss < THRESHOLD_ADJACENCY_DETECTOR and stop_at_threshold:
                print(f"Stopping at epoch {epoch} with loss {epoch_average_loss} because of threshold")
                break

            epoch_average_loss = 0  # Reset for the next average calculation

            pretty_display_reset()
            pretty_display_start(epoch)

    return adjacency_network


def train_adjacency_network_until_threshold(adjacency_network: AdjacencyDetector,
                                            storage: StorageSuperset2) -> AdjacencyDetector:
    adjacency_network = adjacency_network.to(get_device())
    adjacency_network = _train_adjacency_network(adjacency_network, storage, 1501, True, True)

    return adjacency_network


def train_adjacency_network(adjacency_network: AdjacencyDetector,
                            storage: StorageSuperset2) -> AdjacencyDetector:
    adjacency_network = adjacency_network.to(get_device())
    adjacency_network = _train_adjacency_network(adjacency_network, storage, 1501, True)

    return adjacency_network
