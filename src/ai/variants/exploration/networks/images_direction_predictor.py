import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData, calculate_coords_distance
from src.ai.variants.exploration.params import MAX_DISTANCE, ROTATIONS, \
    THRESHOLD_IMAGE_DISTANCE_NETWORK, DIRECTION_THETAS_SIZE
from src.ai.variants.exploration.utils_pure_functions import sample_n_elements, distance_percent_to_distance_thetas
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, pretty_display_set, pretty_display_start, pretty_display_reset
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm


class ImagesDirectionPredictor(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, output_size=DIRECTION_THETAS_SIZE, dropout_rate=0.3,
                 num_blocks=1):
        super(ImagesDirectionPredictor, self).__init__()

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


def direction_loss(image_direction_predictor_network, storage, sample_rate: int = None):
    loss = torch.tensor(0.0)

    connections: RawConnectionData = storage.get_all_connections_only_datapoints_authenticity_filter(
        authentic_distance=True)
    if sample_rate == None:
        sample_rate = len(connections)
    sample_rate = min(sample_rate, len(connections))

    connections: List[RawConnectionData] = sample_n_elements(connections, sample_rate)

    start_embeddings_batch = []
    end_embeddings_batch = []
    target_distances_batch = []

    for connection in connections:
        start = connection["start"]
        end = connection["end"]
        distance = connection["distance"]

        if distance > MAX_DISTANCE:
            distance = MAX_DISTANCE - 0.01

        distance_thetas_target = distance_percent_to_distance_thetas(distance / MAX_DISTANCE,
                                                                     DIRECTION_THETAS_SIZE)
        start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
        end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)

        start_embeddings_batch.append(start_data)
        end_embeddings_batch.append(end_data)
        target_distances_batch.append(distance_thetas_target)

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(get_device())
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(get_device())
    target_distances_batch = torch.stack(target_distances_batch).to(get_device())

    predicted_distances = image_direction_predictor_network.forward_training(start_embeddings_batch,
                                                                             end_embeddings_batch)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    loss = criterion(predicted_distances, target_distances_batch)
    return loss


def _train_images_distance_predictor_network(image_distance_network, storage, num_epochs,
                                             pretty_print=True,
                                             stop_at_threshold: bool = False) -> ImagesDirectionPredictor:
    optimizer = optim.Adam(image_distance_network.parameters(), lr=0.0005, amsgrad=True)

    image_distance_network = image_distance_network.to(get_device())

    epoch_average_loss = 0
    epoch_print_rate = 250

    storage.build_permuted_data_random_rotations_rotation0()
    pretty_display_set(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    if stop_at_threshold:
        num_epochs = 1e7

    for epoch in range(num_epochs):

        pretty_display(epoch % epoch_print_rate)
        epoch_loss = 0.0
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=get_device())

        ITERATIONS = 4
        for i in range(ITERATIONS):
            rand_dir = random.randint(0, ROTATIONS - 1)
            storage.build_permuted_data_random_rotations_rotation_N_with_noise(rand_dir)
            loss += direction_loss(image_distance_network, storage)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / ITERATIONS
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            print("")
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_average_loss:.4f}')

            if epoch_average_loss < THRESHOLD_IMAGE_DISTANCE_NETWORK and stop_at_threshold:
                print(f"Stopping at epoch {epoch} with loss {epoch_average_loss} because of threshold")
                break

            epoch_average_loss = 0  # Reset for the next average calculation
            pretty_display_reset()
            pretty_display_start(epoch)

    return image_distance_network


def train_images_distance_predictor_until_threshold(image_distance_predictor_network: ImagesDirectionPredictor,
                                                    storage: StorageSuperset2) -> ImagesDirectionPredictor:
    storage.build_permuted_data_random_rotations_rotation0()
    image_distance_predictor_network = _train_images_distance_predictor_network(image_distance_predictor_network,
                                                                                storage, 1500, True)
    return image_distance_predictor_network


def train_images_distance_predictor(image_distance_predictor_network: ImagesDirectionPredictor,
                                    storage: StorageSuperset2) -> ImagesDirectionPredictor:
    storage.build_permuted_data_random_rotations_rotation0()
    image_distance_predictor_network = _train_images_distance_predictor_network(image_distance_predictor_network,
                                                                                storage, 5000, True)
    return image_distance_predictor_network
