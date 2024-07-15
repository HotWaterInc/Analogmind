import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.utils import array_to_tensor
from typing import List

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class DirectionNetwork(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(DirectionNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(48, 36),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(36, 12),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(12, 8),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(8, 2),
            nn.Tanh()
        )

    def forward(self, x, y):
        # concatenate the two sensor data
        x = torch.cat((x, y), 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / l2_norm

        return x


def direction_loss(direction_network, sample_rate=64):
    global storage
    loss = torch.tensor(0.0)

    # get connections
    # sample for each connection random rotation
    # pass those to autoencoder, get the output, add to batch
    # add to batch
    # add to other batch expected direction

    datapoints: List[str] = storage.sample_n_random_datapoints(sample_rate)
    start_embeddings_batch = []
    end_embeddings_batch = []

    expected_directions_batch = []

    for datapoint in datapoints:
        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]
            direction = connections_to_point[j]["direction"]

            direction = torch.tensor(direction, dtype=torch.float32, device=device)
            l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
            direction = direction / l2_direction
            direction = direction.to(device)

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)

            start_embedding = autoencoder.encoder_inference(start_data)
            end_embedding = autoencoder.encoder_inference(end_data)
            start_embeddings_batch.append(start_embedding)
            end_embeddings_batch.append(end_embedding)

            expected_directions_batch.append(direction)

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(device)
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(device)
    expected_directions_batch = torch.stack(expected_directions_batch).to(device)

    output_directions_batch = direction_network(start_embeddings_batch, end_embeddings_batch)

    criterion = nn.L1Loss()
    loss = criterion(output_directions_batch, expected_directions_batch)

    return loss


def train_direction_ai(direction_network, num_epochs):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(direction_network.parameters(), lr=0.003)

    scale_direction_loss = 1

    epoch_average_loss = 0
    epoch_print_rate = 100

    storage.build_permuted_data_random_rotations()

    for epoch in range(num_epochs):
        if (epoch % 10 == 0):
            storage.build_permuted_data_random_rotations()

        epoch_loss = 0.0

        optimizer.zero_grad()

        loss = direction_loss(direction_network)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_average_loss:.4f}')
            epoch_average_loss = 0  # Reset for the next average calculation

    return direction_network


def run_tests(direction_network):
    global storage
    ITERATIONS = 10

    total_l1_loss = 0.0
    total_l2_loss = 0.0
    direction_network = direction_network.to(device)

    for i in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()
        datapoints: List[str] = storage.get_all_datapoints()
        output_diffl1 = 0.0
        output_diffl2 = 0.0

        start_embedding_batch = []
        end_embedding_batch = []
        expected_direction_batch = []

        for datapoint in datapoints:
            connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)

            for j in range(len(connections_to_point)):
                start = connections_to_point[j]["start"]
                end = connections_to_point[j]["end"]
                direction = connections_to_point[j]["direction"]

                direction = torch.tensor(direction, dtype=torch.float32, device=device)
                l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
                direction = direction / l2_direction

                start_data = storage.get_datapoint_data_tensor_by_name_permuted(start)
                end_data = storage.get_datapoint_data_tensor_by_name_permuted(end)
                start_embedding = autoencoder.encoder_inference(start_data)
                end_embedding = autoencoder.encoder_inference(end_data)

                start_embedding_batch.append(start_embedding)
                end_embedding_batch.append(end_embedding)
                expected_direction_batch.append(direction)

        start_embedding_batch = torch.stack(start_embedding_batch).to(device)
        end_embedding_batch = torch.stack(end_embedding_batch).to(device)
        direction_pred = direction_network(start_embedding_batch, end_embedding_batch)
        expected_direction_batch = torch.stack(expected_direction_batch).to(device)

        # print(direction_pred)
        # print(expected_direction_batch)

        l1_loss_point = torch.mean(torch.norm(direction_pred - expected_direction_batch, p=1, dim=1))
        l2_loss_point = torch.mean(torch.norm(direction_pred - expected_direction_batch, p=2, dim=1))

        output_diffl1 += l1_loss_point.item()
        output_diffl2 += l2_loss_point.item()

        # print(f"Test {i + 1}/{ITERATIONS} L1 Loss: {output_diffl1:.4f}, L2 Loss: {output_diffl2:.4f}")
        total_l1_loss += output_diffl1
        total_l2_loss += output_diffl2

    total_l1_loss /= ITERATIONS
    total_l2_loss /= ITERATIONS
    print(f"Total L1 Loss: {total_l1_loss:.4f}, Total L2 Loss: {total_l2_loss:.4f}")


def run_new_ai():
    direction_network = DirectionNetwork().to(device)
    train_direction_ai(direction_network, num_epochs=1000)
    save_ai_manually("direction_network_working.pth", direction_network)
    run_tests(direction_network)


def run_loaded_ai():
    direction_network = load_manually_saved_ai("direction_network_working.pth")
    run_tests(direction_network)


def run_direction_network():
    global storage
    global permutor
    global autoencoder

    storage = StorageSuperset2()
    permutor = load_manually_saved_ai("permutor_deshift_working.pth")
    autoencoder = load_manually_saved_ai("autoencodPerm10k_working.pth")

    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()
    storage.tanh_all_data()
    storage.set_permutor(permutor)
    storage.build_permuted_data_raw_with_thetas()

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = None
autoencoder: BaseAutoencoderModel = None
