import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, load_latest_ai, load_manually_saved_ai, \
    save_ai_manually
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage import StorageSuperset
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from typing import List, Dict, Union
from src.utils import array_to_tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ImprovedPermutorAttention2(nn.Module):
    def __init__(self, input_size=24, embedding_size=64, num_heads=8, dropout_rate=0.3):
        super(ImprovedPermutorAttention2, self).__init__()
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
        self.output_layer2 = nn.Linear(embedding_size, input_size)
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


def train_permutor_attention(permutor, epochs):
    global storage
    permutor = permutor.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(permutor.parameters(), lr=0.006)
    epoch_average_loss = 0
    epoch_print_rate = 100

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0).to(device)

        # gets each data batch from storage
        datapoints = storage.sample_n_random_datapoints(64)
        # builds for each one the input and target tensors
        for index, datapoint in enumerate(datapoints):
            name = datapoint
            data_array = storage.get_datapoint_data_tensor_by_name(name)
            thetas_batch = []
            target_batch = []

            for j in range(len(data_array)):
                # creates 36 thetas and makes them 1 proportionally to the index
                length = len(data_array)
                theta_percent = 1 / length * j
                thetas = build_thetas(theta_percent, 36)

                thetas_batch.append(thetas)

                # no its not a bug, we want to "deshift" the data
                target_batch.append(data_array[0])

            thetas_batch = torch.stack(thetas_batch).to(device)
            # print(data_array.shape)  # 20 x 24
            # print(thetas_batch.shape)  # 20 x 36
            data_array = data_array.to(device)
            # Forward pass
            output = permutor(data_array, thetas_batch)
            # target is the first element of the data array for each datapoint
            target = torch.stack(target_batch).to(device)

            loss += criterion(output, target)

        loss /= len(datapoints)
        # loss *= 10
        epoch_average_loss += loss.item()

        loss.backward()
        optimizer.step()

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            print(f"Epoch {epoch}, Loss: {epoch_average_loss}")
            epoch_average_loss = 0

    return permutor


def evaluate_permutor_attention(autoencoder, storage):
    autoencoder = autoencoder.to(device)
    criterion = nn.L1Loss()

    datapoints = storage.get_all_datapoints()

    total_loss = 0
    total_loss_n2 = 0
    for index, datapoint in enumerate(datapoints):
        name = datapoint
        data_array = storage.get_datapoint_data_tensor_by_name(name)

        thetas_batch = []
        target_batch = []
        length = len(data_array)

        for j in range(length):
            # creates 36 thetas and makes them 1 proportionally to the index
            theta_percent = 1 / length * j
            thetas = build_thetas(theta_percent, 36)
            thetas_batch.append(thetas)

            # no its not a bug, we want to "deshift" the data
            target_batch.append(data_array[0])

        thetas_batch = torch.stack(thetas_batch).to(device)
        data_array = data_array.to(device)
        # Forward pass
        output = autoencoder(data_array, thetas_batch)
        # target is the first element of the data array for each datapoint
        target = data_array[0].to(device)

        loss = (torch.norm(output - target, p=1)).mean() / len(data_array)
        loss2 = (torch.norm(output - target, p=2)).mean() / len(data_array)

        total_loss += loss.item()
        total_loss_n2 += loss2.item()

    print("Average loss per datapoint per rotation: (L1)", total_loss / len(datapoints))
    print("Average loss per datapoint per rotation (L2): ", total_loss_n2 / len(datapoints))
    print("For random datapoint from storage: ")
    random_datapoint = random.choice(datapoints)
    name = random_datapoint
    data_array = storage.get_datapoint_data_tensor_by_name(name)

    cap_examples = 5
    for j in range(len(data_array)):
        # creates 36 thetas and makes them 1 proportionally to the index
        length = len(data_array)
        theta_percent = 1 / length * j
        true_theta_index = theta_percent * 36
        thetas = torch.zeros(36)
        decimal = true_theta_index - int(true_theta_index)
        thetas[int(true_theta_index)] = 1 - decimal
        thetas[int(true_theta_index) + 1] = decimal
        # print("index", j, "theta", theta_percent)
        # print("initial data", data_array[j])
        thetas = thetas.to(device)
        data_array = data_array.to(device)
        # Forward pass
        output = autoencoder(data_array[j].unsqueeze(0), thetas.unsqueeze(0))
        # print("output", output)
        if j == cap_examples:
            break


def run_ai():
    # permutor = ImprovedPermutor()
    # trained_permutor = train_permutor3(permutor, epochs=10000)

    permutor = ImprovedPermutorAttention2().to(device)
    trained_permutor = train_permutor_attention(permutor, epochs=2500)
    return trained_permutor


def run_tests(autoencoder):
    global storage
    evaluate_permutor_attention(autoencoder, storage)


def run_new_ai() -> None:
    permutor = run_ai()
    save_ai_manually("permutor_deshift", permutor)
    run_tests(permutor)


def load_ai() -> None:
    global model
    model = load_manually_saved_ai("permutor_deshift_10k.pth")
    run_tests(model)


def run_permutor_deshift() -> None:
    global storage
    storage.load_raw_data_from_others("data8x8_rotated24.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()

    # apply tanh to the entire storage
    def permutorTanh(x):
        return torch.tanh(x)

    storage.set_permutor(permutorTanh)
    storage.build_permuted_data_raw()

    # run_new_ai()
    load_ai()


storage: StorageSuperset2 = StorageSuperset2()
