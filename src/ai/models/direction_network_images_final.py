import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.utils import array_to_tensor
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, set_pretty_display, pretty_display_start, pretty_display_reset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)


class SimpleDirectionNetworkRawAugmented_winputs(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, output_size=4, dropout_rate=0.3, num_blocks=1):
        super(SimpleDirectionNetworkRawAugmented_winputs, self).__init__()

        self.input_layer = nn.Linear(input_size * 2, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, y):
        inpt = torch.cat((x, y), dim=1)

        out = self.input_layer(inpt)
        for block in self.blocks:
            out = block(out)

        return self.output_layer(out)


class DirectionNetworkUpBNRaw(nn.Module):
    def __init__(self, input_size=512, embedding_size=128, num_heads=16, dropout_rate=0.2, output_size=4):
        super(DirectionNetworkUpBNRaw, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size

        # Initial embedding layers
        self.input_embed = nn.Linear(input_size, embedding_size)
        self.bn_input = nn.BatchNorm1d(embedding_size)
        self.input_embed2 = nn.Linear(embedding_size, embedding_size)
        self.bn_input2 = nn.BatchNorm1d(embedding_size)

        self.layers = nn.ModuleList()
        for _ in range(2):
            self.layers.append(nn.ModuleDict({
                'attention': nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads),
                'bn1': nn.BatchNorm1d(embedding_size),
                'fc1': nn.Linear(embedding_size, embedding_size),
                'bn2': nn.BatchNorm1d(embedding_size),
                'fc2': nn.Linear(embedding_size, embedding_size),
                'bn3': nn.BatchNorm1d(embedding_size),
                'tanh': nn.Tanh(),
                'dropout': nn.Dropout(dropout_rate)
            }))

        # Final output layers
        self.output_layer = nn.Linear(embedding_size, embedding_size)
        self.bn_output = nn.BatchNorm1d(embedding_size)
        self.output_layer2 = nn.Linear(embedding_size, output_size)

        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        # Combine input and thetas
        diff = y - x

        # Initial embedding
        x = self.input_embed(diff)
        x = self.bn_input(x)
        x = self.activation(x)
        x = self.input_embed2(x)
        x = self.bn_input2(x)
        x = self.tanh(x)

        # Process through attention and additional layers
        for layer in self.layers:
            residual = x

            # Attention
            x_att = x.unsqueeze(0)  # Add sequence dimension
            x_att, _ = layer['attention'](x_att, x_att, x_att)
            x_att = x_att.squeeze(0)  # Remove sequence dimension
            x = x + x_att  # Add residual connection
            x = layer['bn1'](x)

            # Fully connected layers with dropout
            x = layer['fc1'](x)
            x = layer['bn2'](x)
            x = self.activation(x)
            x = layer['dropout'](x)

            x = layer['fc2'](x)
            x = layer['bn3'](x)
            x = layer['tanh'](x)
            x = layer['dropout'](x)

            # Add residual connection
            x = x + residual

        # Final output
        x = self.output_layer(x)
        x = self.bn_output(x)
        x = self.activation(x)
        x = self.output_layer2(x)

        return x


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


def radians_to_degrees(radians):
    return radians * 180 / np.pi


def embedding_policy(data):
    global autoencoder
    start_embedding = data
    # start_embedding = autoencoder.encoder_inference(data)
    return start_embedding


def direction_loss(direction_network, sample_rate=64):
    global storage
    loss = torch.tensor(0.0)

    # get connections
    # sample for each connection random rotation
    # pass those to autoencoder, get the output, add to batch
    # add to batch
    # add to other batch expected direction

    sample_rate = 25
    datapoints: List[str] = storage.sample_n_random_datapoints(sample_rate)
    start_embeddings_batch = []
    end_embeddings_batch = []

    # expected_directions_batch = []
    target_thetas_batch = []
    targets = []

    count = 0

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

            start_embedding = embedding_policy(start_data)
            end_embedding = embedding_policy(end_data)

            # count += 1
            # if count % 100 == 0:
            #     print(start, end, direction)
            #     # print(start_embedding, end_embedding)

            start_embeddings_batch.append(start_embedding)
            end_embeddings_batch.append(end_embedding)

            direction_x = direction[0].item()
            direction_y = direction[1].item()
            target = None
            if direction_x == 0 and direction_y == 1:
                target = 0
            elif direction_x == -1 and direction_y == 0:
                target = 1
            elif direction_x == 0 and direction_y == -1:
                target = 2
            elif direction_x == 1 and direction_y == 0:
                target = 3

            if target == None:
                raise Exception("Something went wrong with the target")

            targets.append(target)

            # thetas = build_thetas(theta_percent, 36)
            # target_thetas_batch.append(thetas)

            # print(theta_percent, degrees)
            # print(start, end, direction)
            # print(thetas)
            # print(thetas_to_degree(thetas))

    start_embeddings_batch = torch.stack(start_embeddings_batch).to(device)
    end_embeddings_batch = torch.stack(end_embeddings_batch).to(device)

    # target_thetas_batch = torch.stack(target_thetas_batch).to(device)
    # target_thetas_batch = F.normalize(target_thetas_batch, p=1, dim=1)

    # In softmax probability form
    output_thetas_batch = direction_network(start_embeddings_batch, end_embeddings_batch)
    # applies log to the output thetas
    # log_predicted_probs = torch.log_softmax(output_thetas_batch, dim=1, dtype=torch.float32)

    # Define the KL Divergence Loss function
    # criterion = torch.nn.KLDivLoss(reduction='batchmean')
    criterion = torch.nn.CrossEntropyLoss()
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    # Calculate the loss
    # loss = criterion(log_predicted_probs, target_thetas_batch)
    loss = criterion(output_thetas_batch, targets)

    # criterion = nn.L1Loss()
    # loss = criterion(output_thetas_batch, target_thetas_batch)

    return loss


def train_direction_ai(direction_network, num_epochs):
    optimizer = optim.Adam(direction_network.parameters(), lr=0.003, amsgrad=True)

    scale_direction_loss = 1

    epoch_average_loss = 0
    epoch_print_rate = 1000

    storage.build_permuted_data_random_rotations_rotation0()

    set_pretty_display(epoch_print_rate, "Epochs batch training")
    pretty_display_start(0)

    for epoch in range(num_epochs):
        if (epoch % 5 == 0):
            rand_rot = np.random.randint(0, 24)
            # storage.build_permuted_data_random_rotations_rotation0()
            # storage.build_permuted_data_random_rotations_rotation_N(rand_rot)
            # storage.build_permuted_data_random_rotations()

            rot_arr = [rand_rot - 1, rand_rot, rand_rot + 1]

            if rot_arr[0] < 0:
                rot_arr = [23, 0, 1]

            if rot_arr[2] > 23:
                rot_arr = [22, 23, 0]

            storage.build_permuted_data_random_rotations_custom(rot_arr)

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


def thetas_to_degree(thetas):
    length = len(thetas)
    degree = 0
    current_degree = 0

    degree_step = 360 / length
    # take first 2 biggest thetas
    a, b = 0, 0

    for i in range(len(thetas)):
        if thetas[i] > thetas[a]:
            b = a
            a = i
        elif thetas[i] > thetas[b]:
            b = i

    current_degree = a * degree_step
    degree += current_degree * thetas[a]
    current_degree = b * degree_step
    degree += current_degree * thetas[b]

    degree /= (thetas[a] + thetas[b])
    return degree


def run_tests_mini(direction_network):
    global storage, autoencoder

    direction_network = direction_network.to(device)
    direction_network.eval()
    # autoencoder = autoencoder.to(device)

    datapoints: List[str] = storage.get_all_datapoints()

    win = 0
    lose = 0
    ITERATIONS = 10

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
                pred_direction_thetas = F.softmax(pred_direction_thetas, dim=0)

                expected_degree = direction_to_degrees(direction)
                max_pred = torch.argmax(pred_direction_thetas)

                predicted_degree = None
                if max_pred == 0:
                    predicted_degree = 0
                elif max_pred == 1:
                    predicted_degree = 90
                elif max_pred == 2:
                    predicted_degree = 180
                elif max_pred == 3:
                    predicted_degree = 270

                if predicted_degree == expected_degree:
                    win += 1
                else:
                    error_arr.append((start, end, index_start, index_end))
                    lose += 1

    print("")
    # print(error_arr)
    print("Win", win)
    print("Lose", lose)
    print("Win rate", win / (win + lose))

    # start_embedding_batch = torch.stack(start_embedding_batch).to(device)
    # end_embedding_batch = torch.stack(end_embedding_batch).to(device)
    # direction_pred = direction_network(start_embedding_batch, end_embedding_batch)
    # expected_direction_batch = torch.stack(expected_direction_batch).to(device)

    # print(direction_pred)
    # print(expected_direction_batch)


def run_new_ai():
    # direction_network = SimpleDirectionNetworkRawAugmented().to(device)
    direction_network = SimpleDirectionNetworkRawAugmented_winputs().to(device)

    train_direction_ai(direction_network, num_epochs=10000)
    save_ai_manually("direction", direction_network)
    run_tests_mini(direction_network)
    # run_tests(direction_network)


def run_loaded_ai():
    direction_network = load_manually_saved_ai("direction_image_raw.pth")

    run_tests_mini(direction_network)
    # run_tests(direction_network)


def run_direction_network_images_final():
    global storage
    global permutor
    global autoencoder

    storage = StorageSuperset2()
    # autoencoder = load_manually_saved_ai("autoencod_images_full_without_recons.pth")

    grid_data = 5
    storage.load_raw_data_from_others(f"data{grid_data}x{grid_data}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_data}x{grid_data}_connections.json")

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = None
autoencoder: BaseAutoencoderModel = None
