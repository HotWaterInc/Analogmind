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


class ImprovedPermutorAttention(nn.Module):
    def __init__(self, input_size=24, embedding_size=64, num_heads=8):
        super(ImprovedPermutorAttention, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size

        # Initial embedding layer
        self.input_embed = nn.Linear(input_size + 36, embedding_size)

        # Three attention layers, each followed by 3 fully connected layers
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(nn.ModuleDict({
                'attention': nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads),
                'fc1': nn.Linear(embedding_size, embedding_size),
                'fc2': nn.Linear(embedding_size, embedding_size),
                'fc3': nn.Linear(embedding_size, embedding_size)
            }))

        # Final output layer
        self.output_layer = nn.Linear(embedding_size, input_size)

        self.activation = nn.LeakyReLU()

    def forward(self, x, thetas):
        # Combine input and thetas
        x = torch.cat([x, thetas], dim=1)

        # Initial embedding
        x = self.input_embed(x)
        x = self.activation(x)

        # Process through attention and FC layers
        for layer in self.layers:
            # Attention
            x_att = x.unsqueeze(0)  # Add sequence dimension
            x_att, _ = layer['attention'](x_att, x_att, x_att)
            x_att = x_att.squeeze(0)  # Remove sequence dimension

            # Fully connected layers
            x = self.activation(layer['fc1'](x_att))
            x = self.activation(layer['fc2'](x))
            x = self.activation(layer['fc3'](x))

        # Final output
        x = self.output_layer(x)

        return x


def train_permutor_attention(permutor, epochs):
    global storage
    permutor = permutor.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(permutor.parameters(), lr=0.01)
    epoch_average_loss = 0
    epoch_print_rate = 100

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0).to(device)

        # gets each data batch from storage
        datapoints = storage.sample_n_random_datapoints(32)
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
                true_theta_index = theta_percent * 36
                thetas = torch.zeros(36)
                decimal = true_theta_index - int(true_theta_index)
                thetas[int(true_theta_index)] = 1 - decimal
                thetas[int(true_theta_index) + 1] = decimal
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
            target = data_array[0].to(device)

            loss += criterion(output, target)

        loss /= len(datapoints)
        epoch_average_loss += loss.item()

        loss.backward()
        optimizer.step()

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            print(f"Epoch {epoch}, Loss: {epoch_average_loss}")
            epoch_average_loss = 0

    return permutor


def run_ai():
    # permutor = ImprovedPermutor()
    # trained_permutor = train_permutor3(permutor, epochs=10000)

    permutor = ImprovedPermutorAttention().to(device)
    trained_permutor = train_permutor_attention(permutor, epochs=5000)
    return trained_permutor


def run_tests(autoencoder):
    global storage
    # evaluate_permutor(autoencoder, storage)


def run_new_ai() -> None:
    permutor = run_ai()
    save_ai_manually("permutor10k_lrn", permutor)
    run_tests(permutor)


def load_ai() -> None:
    global model
    model = load_manually_saved_ai("permutor10k.pth")
    run_tests(model)


def run_permutor_deshift() -> None:
    global storage
    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()

    run_new_ai()
    # load_ai()


storage: StorageSuperset = StorageSuperset()
