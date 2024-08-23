import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData
from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.params import THETAS_SIZE

from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.utils import array_to_tensor
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, pretty_display_set, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import thetas_to_radians, \
    angle_percent_to_thetas_normalized_cached, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm


class DirectionNetworkThetas(nn.Module):
    def __init__(self, input_size=128, hidden_size=512, output_size=THETAS_SIZE, dropout_rate=0.3, num_blocks=1):
        super(DirectionNetworkThetas, self).__init__()

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


def embedding_policy(data):
    global autoencoder
    start_embedding = data
    # start_embedding = autoencoder.encoder_inference(data)
    return start_embedding


def direction_loss(direction_network, sample_rate=25):
    global storage
    loss = torch.tensor(0.0)

    datapoints: List[str] = storage.sample_n_random_datapoints(sample_rate)
    start_embeddings_batch = []
    end_embeddings_batch = []

    # expected_directions_batch = []
    target_thetas_batch = []

    counter = 0
    for datapoint in datapoints:
        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]
            direction = connections_to_point[j]["direction"]
            direction = torch.tensor(direction, dtype=torch.float32)

            final_radian = coordinate_pair_to_radians_cursed_tranform(direction[0], direction[1])
            radian_percent = radians_to_percent(final_radian)
            thetas_target = angle_percent_to_thetas_normalized_cached(radian_percent, 36)

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)

            start_embedding = embedding_policy(start_data)
            end_embedding = embedding_policy(end_data)

            start_embeddings_batch.append(start_embedding)
            end_embeddings_batch.append(end_embedding)

            target_thetas_batch.append(thetas_target)

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(device)
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(device)

    target_thetas_batch = torch.stack(target_thetas_batch).to(device)

    # Log softmax output
    output_thetas_batch = direction_network.forward_training(start_embeddings_batch, end_embeddings_batch)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    loss = criterion(output_thetas_batch, target_thetas_batch)
    return loss


def train_direction_ai(direction_network, num_epochs):
    optimizer = optim.Adam(direction_network.parameters(), lr=0.0005, amsgrad=True)

    scale_direction_loss = 1

    epoch_average_loss = 0
    epoch_print_rate = 100

    storage.build_permuted_data_random_rotations_rotation0()

    pretty_display_set(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    for epoch in range(num_epochs):
        if (epoch % 5 == 0):
            rand_rot = np.random.randint(0, 24)
            # storage.build_permuted_data_random_rotations_rotation0()
            # storage.build_permuted_data_random_rotations_rotation_N(rand_rot)
            storage.build_permuted_data_random_rotations()

        pretty_display(epoch % epoch_print_rate)

        epoch_loss = 0.0

        optimizer.zero_grad()

        loss = direction_loss(direction_network) * scale_direction_loss
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


# def thetas_to_degree(thetas):
#     length = len(thetas)
#     degree = 0
#     current_degree = 0
#
#     degree_step = 360 / length
#     # take first 2 biggest thetas
#     a, b = 0, 0
#
#     for i in range(len(thetas)):
#         if thetas[i] > thetas[a]:
#             b = a
#             a = i
#         elif thetas[i] > thetas[b]:
#             b = i
#
#     current_degree = a * degree_step
#     degree += current_degree * thetas[a]
#     current_degree = b * degree_step
#     degree += current_degree * thetas[b]
#
#     degree /= (thetas[a] + thetas[b])
#     return degree


def direction_to_degrees(direction):
    y = direction[1]
    x = direction[0]
    degrees = None

    if y == 1:
        degrees = 0
    elif y == -1:
        degrees = 180
    elif x == 1:
        degrees = 270
    elif x == -1:
        degrees = 90

    if degrees == None:
        raise Exception("Shit went wrong as always dir to deg")

    return degrees


def run_tests_permuted_data_unseen_before(direction_network):
    global storage, autoencoder

    direction_network = direction_network.to(device)
    direction_network.eval()
    autoencoder = autoencoder.to(device)

    datapoints: List[str] = storage.get_all_datapoints()

    win = 0
    lose = 0
    ITERATIONS = 5

    error_arr = []
    for iter in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()
        for datapoint in datapoints:
            connections_to_point: List[
                RawConnectionData] = storage.get_datapoints_adjacent_at_degree_n_as_raw_connection_data(datapoint, 2)

            for j in range(len(connections_to_point)):
                start = connections_to_point[j]["start"]
                end = connections_to_point[j]["end"]
                direction = connections_to_point[j]["direction"]

                direction = torch.tensor(direction, dtype=torch.float32, device=device)
                l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
                direction = direction / l2_direction

                start_data = storage.get_datapoint_data_tensor_by_name_permuted(start).to(device)
                end_data = storage.get_datapoint_data_tensor_by_name_permuted(end).to(device)
                metadata = storage.get_pure_permuted_raw_env_metadata_array_rotation()
                index_start = storage.get_datapoint_index_by_name(start)
                index_end = storage.get_datapoint_index_by_name(end)

                start_embedding = embedding_policy(start_data).unsqueeze(0)
                end_embedding = embedding_policy(end_data).unsqueeze(0)

                pred_direction_thetas = direction_network(start_embedding, end_embedding).squeeze(0)

                predicted_degree = radians_to_degrees(thetas_to_radians(pred_direction_thetas))
                expected_degree = direction_to_degrees_atan(direction)
                # expected_degree = direction_to_degrees(direction)

                if math.fabs(predicted_degree - expected_degree) < 15:
                    win += 1
                else:
                    lose += 1

    print("")
    print("Win", win)
    print("Lose", lose)
    print("Win rate", win / (win + lose))

    # start_embedding_batch = torch.stack(start_embedding_batch).to(device)
    # end_embedding_batch = torch.stack(end_embedding_batch).to(device)
    # direction_pred = direction_network(start_embedding_batch, end_embedding_batch)
    # expected_direction_batch = torch.stack(expected_direction_batch).to(device)

    # print(direction_pred)
    # print(expected_direction_batch)


def run_tests_permuted_data(direction_network):
    global storage, autoencoder

    direction_network = direction_network.to(device)
    direction_network.eval()
    autoencoder = autoencoder.to(device)

    datapoints: List[str] = storage.get_all_datapoints()

    win = 0
    lose = 0
    ITERATIONS = 5

    error_arr = []
    for iter in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()
        for datapoint in datapoints:
            connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)

            for j in range(len(connections_to_point)):
                start = connections_to_point[j]["start"]
                end = connections_to_point[j]["end"]
                direction = connections_to_point[j]["direction"]

                direction = torch.tensor(direction, dtype=torch.float32, device=device)
                l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
                direction = direction / l2_direction

                start_data = storage.get_datapoint_data_tensor_by_name_permuted(start).to(device)
                end_data = storage.get_datapoint_data_tensor_by_name_permuted(end).to(device)
                metadata = storage.get_pure_permuted_raw_env_metadata_array_rotation()
                index_start = storage.get_datapoint_index_by_name(start)
                index_end = storage.get_datapoint_index_by_name(end)

                start_embedding = embedding_policy(start_data).unsqueeze(0)
                end_embedding = embedding_policy(end_data).unsqueeze(0)

                pred_direction_thetas = direction_network(start_embedding, end_embedding).squeeze(0)

                predicted_degree = radians_to_degrees(thetas_to_radians(pred_direction_thetas))
                # expected_degree = direction_to_degrees(direction)
                expected_degree = direction_to_degrees_atan(direction)
                # print("Predicted degree", predicted_degree)

                if math.fabs(predicted_degree - expected_degree) < 1:
                    win += 1
                else:
                    lose += 1

    print("")
    print("Win", win)
    print("Lose", lose)
    print("Win rate", win / (win + lose))

    # start_embedding_batch = torch.stack(start_embedding_batch).to(device)
    # end_embedding_batch = torch.stack(end_embedding_batch).to(device)
    # direction_pred = direction_network(start_embedding_batch, end_embedding_batch)
    # expected_direction_batch = torch.stack(expected_direction_batch).to(device)

    # print(direction_pred)
    # print(expected_direction_batch)


def run_tests(direction_network):
    run_tests_permuted_data(direction_network)
    run_tests_permuted_data_unseen_before(direction_network)


def run_new_ai():
    direction_network = DirectionNetworkThetas().to(device)
    direction_network = train_direction_ai(direction_network, num_epochs=2001)
    save_ai_manually("direction_SS_thetas_postautoencod", direction_network)
    run_tests(direction_network)


def run_loaded_ai():
    direction_network = load_manually_saved_ai("direction_SS_thetas_postautoencod_saved.pth")
    run_tests(direction_network)


def storage_to_manifold():
    global storage
    global autoencoder
    autoencoder = load_manually_saved_ai("camera1_full_forced_saved.pth")
    autoencoder.eval()
    autoencoder = autoencoder.to(device)

    storage.set_permutor(autoencoder)
    storage.build_permuted_data_raw_abstraction_autoencoder_manifold()


def run_direction_post_autoencod_SS():
    global storage
    global autoencoder

    storage = StorageSuperset2()
    autoencoder = load_manually_saved_ai("camera1_full_forced_saved.pth")

    grid_data = 5
    storage.load_raw_data_from_others(f"data{grid_data}x{grid_data}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_data}x{grid_data}_connections.json")
    storage_to_manifold()

    # run_new_ai()
    run_loaded_ai()


storage: StorageSuperset2 = None
autoencoder: BaseAutoencoderModel = None

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
