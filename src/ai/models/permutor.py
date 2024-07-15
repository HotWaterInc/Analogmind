import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, load_latest_ai, load_manually_saved_ai, \
    save_ai_manually
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage import StorageSuperset
from typing import List, Dict, Union
from src.utils import array_to_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PermutorFinal1(nn.Module):
    def __init__(self, input_size=24, embedding_size=64, num_heads=8, dropout_rate=0.2, output_size=8):
        super(PermutorFinal1, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size

        # Initial embedding layer
        self.input_embed = nn.Linear(input_size + 36, embedding_size)

        # Three attention layers, each followed by 1 CNN and 2 FNN
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(nn.ModuleDict({
                'attention': nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads),
                'cnn': nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1),
                'fc1': nn.Linear(embedding_size, embedding_size),
                'fc2': nn.Linear(embedding_size, embedding_size),
                'tanh': nn.Tanh(),
                'dropout': nn.Dropout(dropout_rate)
            }))

        # Final output layers
        self.output_layer = nn.Linear(embedding_size, embedding_size)
        self.output_layer2 = nn.Linear(embedding_size, output_size)
        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, thetas):
        # Combine input and thetas
        x = torch.cat([x, thetas], dim=1)

        # Initial embedding
        x = self.input_embed(x)
        x = self.activation(x)

        # Process through attention and additional layers
        for layer in self.layers:
            residual = x

            # Attention
            x_att = x.unsqueeze(0)  # Add sequence dimension
            x_att, _ = layer['attention'](x_att, x_att, x_att)
            x_att = x_att.squeeze(0)  # Remove sequence dimension

            # CNN
            x = x_att.unsqueeze(2)  # Add feature dimension for CNN
            x = layer['cnn'](x)
            x = x.squeeze(2)  # Remove feature dimension
            x = self.activation(x)

            # Fully connected layers with dropout
            x = self.activation(layer['fc1'](x))
            x = layer['dropout'](x)
            x = layer['tanh'](layer['fc2'](x))  # Add tanh to one of the FNNs
            x = layer['dropout'](x)

            # Add residual connection
            x = x + residual

        # Final output
        x = self.output_layer(x)
        x = self.tanh(x)
        x = self.output_layer2(x)
        x = self.tanh(x)

        return x


import math
from scipy.stats import norm


def build_thetas(true_theta, thetas_length):
    thetas = torch.zeros(thetas_length)
    true_theta_index = true_theta * (thetas_length - 1)
    integer_index = int(true_theta_index)

    FILL_DISTANCE = 4
    for i in range(FILL_DISTANCE):
        left_index = integer_index - i
        right_index = integer_index + i

        pdf_value = norm.pdf(left_index, loc=true_theta_index, scale=1)
        if left_index < 0:
            left_index = len(thetas) + left_index
        thetas[left_index] = pdf_value

        pdf_value = norm.pdf(right_index, loc=true_theta_index, scale=1)
        if right_index >= len(thetas):
            right_index = right_index - len(thetas)
        thetas[right_index] = pdf_value

    # Normalize thetas so the maximum value is 1
    sd = 1
    peak_value = 1 / (sd * math.sqrt(2 * math.pi))
    thetas /= peak_value
    return thetas


def rotations_distance_loss(permutor, batch_size) -> torch.Tensor:
    global storage
    datapoints_names: List[str] = storage.sample_n_random_datapoints(batch_size)
    datapoints_data: List[torch.Tensor] = [storage.get_datapoint_data_tensor_by_name(datapoint_name).to(device) for
                                           datapoint_name in datapoints_names]
    # datapoints_data = [datapoint_data.unsqueeze(1) for datapoint_data in datapoints_data]
    accumulated_loss = torch.tensor(0.0, device=device)  # Create tensor directly on the device
    # build 20, 36 thetas for each rotation
    thetas_batch = []
    for i in range(len(datapoints_data[0])):
        thetas = build_thetas(i / len(datapoints_data[0]), 36)
        thetas_batch.append(thetas)

    thetas_batch = torch.stack(thetas_batch).to(device)

    for datapoint_data in datapoints_data:
        datapoint_data = datapoint_data.to(device)  # Assign the result back to datapoint_data
        outputs: torch.Tensor = permutor(datapoint_data, thetas_batch)
        pairwise_distances = torch.cdist(outputs, outputs, p=2)
        accumulated_loss += torch.sum(pairwise_distances)

    accumulated_loss /= batch_size
    return accumulated_loss


def datapoint_distance_loss(permutor, non_adjacent_sample_size: int, distance_factor: float) -> torch.Tensor:
    sampled_pairs = storage.sample_datapoints_adjacencies(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []

    batch_thetas1 = []
    batch_thetas2 = []

    for pair in sampled_pairs:
        datapoint1, index1 = storage.get_datapoint_data_random_rotation_tensor_by_name_and_index(pair["start"])
        datapoint2, index2 = storage.get_datapoint_data_random_rotation_tensor_by_name_and_index(pair["end"])

        datapoint1_alldata = storage.get_datapoint_data_tensor_by_name(pair["start"])
        # gets the number of rotations in each datapoint
        rotations_number = len(datapoint1_alldata)
        index1 = index1 / rotations_number
        index2 = index2 / rotations_number

        thetas_datapoint1 = build_thetas(index1 / rotations_number, 36)
        thetas_datapoint2 = build_thetas(index2 / rotations_number, 36)

        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)
        batch_thetas1.append(thetas_datapoint1)
        batch_thetas2.append(thetas_datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1).to(device)
    batch_datapoint2 = torch.stack(batch_datapoint2).to(device)
    batch_thetas1 = torch.stack(batch_thetas1).to(device)
    batch_thetas2 = torch.stack(batch_thetas2).to(device)

    encoded_i = permutor(batch_datapoint1, batch_thetas1)
    encoded_j = permutor(batch_datapoint2, batch_thetas2)

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)

    expected_distance = [pair["distance"] * distance_factor for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance, device=device)  # Move to device

    datapoint_distances_loss = (distance - expected_distance) / distance_factor
    datapoint_distances_loss = torch.square(datapoint_distances_loss)
    datapoint_distances_loss = torch.mean(datapoint_distances_loss)
    return datapoint_distances_loss.to(device)  # Ensure the final result is on the device


def train_permutor_attention(permutor, epochs):
    global storage

    optimizer = optim.Adam(permutor.parameters(), lr=0.001)
    BATCH_SIZE = 64

    DISTANCE_THRESHOLD = 0.4

    scale_datapoint_loss = 0.5
    scale_rotation_loss = 0.02

    datapoints_adjacent_sample = 100

    epoch_print_rate = 10

    average_rotation_loss = 0
    average_datapoint_loss = 0
    average_epoch_loss = 0

    for epoch in range(epochs):
        # LOSS SIMILAR ROTATIONS
        # selects batch size of datapoints

        rotation_loss = torch.tensor(0.0)
        rotation_loss = rotations_distance_loss(permutor, BATCH_SIZE) * scale_rotation_loss
        rotation_loss.backward()
        # LOSS ADJACENT DATAPOINTS

        datapoint_loss = torch.tensor(0.0)
        # datapoint_loss = datapoint_distance_loss(permutor, datapoints_adjacent_sample,
        #                                          DISTANCE_THRESHOLD) * scale_datapoint_loss
        # datapoint_loss.backward()

        optimizer.step()

        average_rotation_loss += rotation_loss.item()
        average_datapoint_loss += datapoint_loss.item()
        average_epoch_loss += rotation_loss.item() + datapoint_loss.item()
        if epoch % epoch_print_rate == 0:
            average_rotation_loss /= epoch_print_rate
            average_datapoint_loss /= epoch_print_rate
            average_epoch_loss /= epoch_print_rate

            print(f"Epoch {epoch}")
            print(f"ROTATION LOSS: {average_rotation_loss} DATASET LOSS: {average_datapoint_loss}")
            print(f"AVERAGE LOSS: {average_epoch_loss}")

            average_rotation_loss = 0
            average_datapoint_loss = 0
            average_epoch_loss = 0
    return permutor


def run_ai():
    # permutor = ImprovedPermutor()
    # trained_permutor = train_permutor3(permutor, epochs=10000)

    permutor = PermutorFinal1().to(device)
    trained_permutor = train_permutor_attention(permutor, epochs=5)
    return trained_permutor


def evaluate_permutor(permutor: PermutorFinal1, storage: Storage) -> None:
    # Move permutor to device if it's not already there
    permutor = permutor.to(device)

    # evaluate difference between permutations ( should be close to 0 )
    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data())).to(device)
    norm_sum = torch.tensor(0.0, device=device)
    count = 0
    permutation_distance_array = []
    datapoint_means = []
    datapoint_outputs_array = []

    for datapoint in train_data:
        datapoint_outputs = []
        for index, rotation in enumerate(datapoint):
            thetas = build_thetas(index / len(rotation), 36)

            output = permutor(rotation, thetas)
            output = output
            datapoint_outputs.append(output)

        datapoint_outputs_array.append(datapoint_outputs)

        for i in range(len(datapoint_outputs)):
            for j in range(i + 1, len(datapoint_outputs)):
                count += 1
                norm_sum += torch.norm(datapoint_outputs[i] - datapoint_outputs[j], p=2)

        randi = random.randint(0, len(datapoint_outputs) - 1)
        randj = random.randint(0, len(datapoint_outputs) - 1)
        permutation_distance_array.append(torch.norm(datapoint_outputs[randi] - datapoint_outputs[randj], p=2).item())

        datapoint_mean = torch.mean(torch.stack(datapoint_outputs), dim=0)
        datapoint_means.append(datapoint_mean)

    print("Permutations distance: ", (norm_sum / count).item())

    # evaluate difference between adjacent points ( should be close to expected distance, much,much farther than the distance between permutations )
    sampled_connections = storage.get_all_adjacent_data()
    avg_distances = {}

    for pair in sampled_connections:
        start_name = pair["start"]
        end_name = pair["end"]
        distance = pair["distance"]
        datapoint1_index = storage.get_datapoint_data_tensor_index_by_name(start_name)
        datapoint2_index = storage.get_datapoint_data_tensor_index_by_name(end_name)
        datapoint1 = datapoint_means[datapoint1_index]
        datapoint2 = datapoint_means[datapoint2_index]

        if f"{distance}" not in avg_distances:
            avg_distances[f"{distance}"] = {
                "sum": torch.tensor(0.0, device=device),
                "count": 0
            }

        distance_between_embeddings = torch.norm(datapoint1 - datapoint2, p=2)
        avg_distances[f"{distance}"]["sum"] += distance_between_embeddings
        avg_distances[f"{distance}"]["count"] += 1

    for distance in avg_distances:
        avg_distances[distance]["sum"] /= avg_distances[distance]["count"]
        print(f"Average distance for distance {distance}: {avg_distances[distance]['sum'].item():.4f}")


def run_tests(autoencoder):
    global storage
    evaluate_permutor(autoencoder, storage)


def run_new_ai() -> None:
    permutor = run_ai()
    save_ai_manually("permutor10k_lrn", permutor)
    run_tests(permutor)


def load_ai() -> None:
    global model
    model = load_manually_saved_ai("permutor10k.pth")
    run_tests(model)


def run_permutor() -> None:
    global storage
    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()
    storage.tanh_all_data()

    run_new_ai()
    # load_ai()


storage: StorageSuperset = StorageSuperset()
