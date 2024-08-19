import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually, load_custom_ai, \
    load_other_ai
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.utils import array_to_tensor
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, set_pretty_display, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import angle_to_thetas, thetas_to_radians, \
    angle_percent_to_thetas_normalized, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan, degrees_to_percent
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm

THETAS_SIZE = 36
MANIFOLD_SIZE = 128


class DirectionNetworkSDS(nn.Module):
    def __init__(self, manifold_size=MANIFOLD_SIZE, direction_thetas_size=THETAS_SIZE, hidden_size=512,
                 dropout_rate=0.3,
                 num_blocks=1):
        super(DirectionNetworkSDS, self).__init__()

        self.input_layer = nn.Linear(manifold_size + direction_thetas_size, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, manifold_size)

    def _forward_pass(self, x, y):
        inpt = torch.cat((x, y), dim=1)

        out = self.input_layer(inpt)
        for block in self.blocks:
            out = block(out)

        output = self.output_layer(out)
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


def SDS_loss(direction_network, sample_rate=25):
    global storage_raw
    loss = torch.tensor(0.0)

    datapoints: List[str] = storage.sample_n_random_datapoints(sample_rate)

    start_embeddings_batch = []
    direction_thetas_batch = []

    target_embeddings_batch = []

    for datapoint in datapoints:
        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]
            direction = connections_to_point[j]["direction"]
            direction = torch.tensor(direction, dtype=torch.float32)
            direction_normalized = direction / torch.norm(direction, p=2, dim=0, keepdim=True)

            direction_angle = direction_to_degrees_atan(direction_normalized)
            direction_percent = degrees_to_percent(direction_angle)
            theta_form = angle_percent_to_thetas_normalized(direction_percent, 36)

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)
            start_embedding = embedding_policy(start_data)
            end_embedding = embedding_policy(end_data)

            start_embeddings_batch.append(start_embedding)
            direction_thetas_batch.append(theta_form)

            target_embeddings_batch.append(end_embedding)

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(device)
    direction_thetas_batch = torch.stack(direction_thetas_batch).to(device)

    target_embeddings_batch = torch.stack(target_embeddings_batch).to(device)
    predicted_embeddings_batch = direction_network.forward_training(start_embeddings_batch, direction_thetas_batch)

    criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()

    loss = criterion(predicted_embeddings_batch, target_embeddings_batch)
    loss += criterion2(predicted_embeddings_batch, target_embeddings_batch)

    return loss


def train_direction_SDS(direction_network, num_epochs):
    optimizer = optim.Adam(direction_network.parameters(), lr=0.005, amsgrad=True)

    scale_direction_loss = 10

    epoch_average_loss = 0
    epoch_print_rate = 100

    storage_raw.build_permuted_data_random_rotations_rotation0()

    set_pretty_display(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    for epoch in range(num_epochs):
        if (epoch % 5 == 0):
            storage_raw.build_permuted_data_random_rotations()

        pretty_display(epoch % epoch_print_rate)

        epoch_loss = 0.0

        optimizer.zero_grad()

        loss = SDS_loss(direction_network) * scale_direction_loss
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


def normalize_direction(direction):
    direction = torch.tensor(direction, dtype=torch.float32, device=device)
    l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
    direction = direction / l2_direction
    return direction


def run_tests_basic(direction_network_SDS):
    global storage_raw, autoencoder

    direction_network_SDS = direction_network_SDS.to(device)
    direction_network_SDS.eval()
    autoencoder = autoencoder.to(device)

    datapoints: List[str] = storage.get_all_datapoints()

    average_error = 0
    ITERATIONS = 5

    error_arr = []
    for iter in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()
        for datapoint in datapoints:
            connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
            error_datapoint = 0

            for j in range(len(connections_to_point)):
                start = connections_to_point[j]["start"]
                end = connections_to_point[j]["end"]
                direction = connections_to_point[j]["direction"]

                direction = normalize_direction(direction)
                direction_angle = direction_to_degrees_atan(direction)
                direction_thetas = angle_percent_to_thetas_normalized(degrees_to_percent(direction_angle), THETAS_SIZE)

                start_data = storage.get_datapoint_data_tensor_by_name_permuted(start).to(device)
                end_data = storage.get_datapoint_data_tensor_by_name_permuted(end).to(device)

                start_manifold = embedding_policy(start_data).unsqueeze(0)
                end_manifold = embedding_policy(end_data).unsqueeze(0)
                direction_thetas = direction_thetas.unsqueeze(0).to(device)

                predicted_manifold = direction_network_SDS(start_manifold, direction_thetas).squeeze(0)
                expected_manifold = end_manifold.squeeze(0)

                # loss
                error_datapoint += torch.norm(predicted_manifold - expected_manifold, p=2, dim=0,
                                              keepdim=True).item()

            average_error += error_datapoint / len(connections_to_point)

    print("")
    print(f"Average error: {average_error / (ITERATIONS * len(datapoints))}")


def run_tests(direction_network):
    run_tests_basic(direction_network)


def run_new_ai():
    direction_network = DirectionNetworkSDS().to(device)
    direction_network = train_direction_SDS(direction_network, num_epochs=1001)
    save_ai_manually("direction_SDS", direction_network)
    run_tests(direction_network)


def run_loaded_ai():
    direction_network = load_custom_ai("")
    run_tests(direction_network)


def storage_to_manifold():
    global storage_raw
    global autoencoder
    autoencoder = load_custom_ai(AUTOENCODER_NAME, MODELS_FOLDER)
    autoencoder.eval()
    autoencoder = autoencoder.to(device)

    storage.set_permutor(autoencoder)
    storage.build_permuted_data_raw_abstraction_autoencoder_manifold()


def run_direction_post_autoencod_SDS():
    global storage_raw
    global autoencoder

    storage = StorageSuperset2()
    grid_data = 5
    storage.load_raw_data_from_others(f"data{grid_data}x{grid_data}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_data}x{grid_data}_connections.json")
    storage_to_manifold()

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = None
autoencoder: BaseAutoencoderModel = None

MODELS_FOLDER = "models_v11"
AUTOENCODER_NAME = "camera1_autoencoder_v1.1.pth"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
