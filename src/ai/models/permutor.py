import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, load_latest_ai, load_manually_saved_ai, \
    save_ai_manually
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage import StorageSuperset
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PermutorRawDNN(nn.Module):
    def __init__(self):
        super(PermutorRawDNN, self).__init__()

        dim1 = 24
        dim2 = 100
        dim3 = 75
        dim4 = 50
        dim5 = 25
        dim6 = 8
        self.fc1 = nn.Linear(dim1, dim2)
        self.actfc1 = nn.Tanh()
        self.fc2 = nn.Linear(dim2, dim3)
        self.actfc2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(dim3, dim4)
        self.actfc3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(dim4, dim5)
        self.actfc4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(dim5, dim6)
        self.actfc5 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.actfc1(x)

        x = self.fc2(x)
        x = self.actfc2(x)

        x = self.fc3(x)
        x = self.actfc3(x)

        x = self.fc4(x)
        x = self.actfc4(x)

        x = self.fc5(x)
        x = self.actfc5(x)
        return x


class Permutor(nn.Module):
    def __init__(self):
        super(Permutor, self).__init__()

        # 1D CNN layers for processing 1D inputs
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.activation1 = nn.LeakyReLU()

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=5, padding=2)
        self.activation2 = nn.LeakyReLU()

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.activation3 = nn.LeakyReLU()

        length = 24
        output_channels = 8
        dim1 = length * output_channels
        dim2 = 100
        dim3 = 50
        dim4 = 8
        self.fc1 = nn.Linear(dim1, dim2)
        self.actfc1 = nn.Tanh()
        self.fc2 = nn.Linear(dim2, dim3)
        self.actfc2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(dim3, dim4)
        self.actfc3 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        x = self.activation3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.actfc1(x)

        x = self.fc2(x)
        x = self.actfc2(x)

        x = self.fc3(x)
        x = self.actfc3(x)

        return x


class ImprovedPermutor(nn.Module):
    def __init__(self, input_size=24, embedding_size=8):
        super(ImprovedPermutor, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # Calculate the correct input size for the fc layer
        self.fc_input_size = 64 * (input_size + 6)  # +6 because of the circular padding
        self.fc = nn.Linear(self.fc_input_size, embedding_size)

        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Ensure input is 3D: (batch_size, channels, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply circular padding
        x = torch.cat([x[:, :, -3:], x, x[:, :, :3]], dim=2)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        x = x.view(x.size(0), -1)

        # Print shape for debugging

        x = self.fc(x)
        # x = self.sigmoid(x)
        x = F.normalize(x, p=2, dim=1)

        return x


def train_permutor2(permutor, train_data, epochs=2000, batch_size=32):
    optimizer = optim.Adam(permutor.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    epoch_print_rate = 1000

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            anchors = []
            positives = []
            negatives = []

            for datapoint in batch:
                # Randomly select two rotations of the same datapoint
                anchor, positive = random.sample(list(datapoint), 2)

                # Randomly select a rotation from a different datapoint
                negative = random.choice(random.choice([dp for dp in train_data if dp is not datapoint]))

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

            anchors = torch.stack(anchors).unsqueeze(1)
            positives = torch.stack(positives).unsqueeze(1)
            negatives = torch.stack(negatives).unsqueeze(1)

            anchor_out = permutor(anchors)
            positive_out = permutor(positives)
            negative_out = permutor(negatives)

            loss = criterion(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % epoch_print_rate == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    return permutor


def reconstruction_handling(autoencoder: Permutor, data: any, criterion: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    enc = autoencoder.encoder_training(data)
    dec = autoencoder.decoder_training(enc)
    return criterion(dec, data) * scale_reconstruction_loss


def train_permutor(permutor: any, epochs: int) -> any:
    # PARAMETERS
    criterion = nn.L1Loss()
    optimizer = optim.Adam(permutor.parameters(), lr=0.02)

    num_epochs = epochs

    scale_rotation_adjacency_loss = 10
    # scale_rotation_adjacency_loss = 0

    scale_datapoint_loss = 0.2
    sampled_datapoints_count = 50
    DATAPOINTS_DISTANCE_FACTOR = 0.1

    epoch_average_loss = 0
    rotation_average_loss = 0
    datapoint_average_loss = 0

    epoch_print_rate = 250
    sample_rotations_pairs_count = 50

    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data()))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        datapoint_means = []

        rotation_adjacency_loss = torch.tensor(0.0)
        datapoint_loss = torch.tensor(0.0)
        # get the mean of all representations
        for datapoint in train_data:
            datapoint_outputs = []

            for rotation in datapoint:
                output = permutor(rotation.unsqueeze(0).unsqueeze(0))
                output = output.squeeze(0)
                datapoint_outputs.append(output)

            # SIMILARITY LOSS FOR SAME PERMUTATIONS
            # sample 50 random pairs of datapoints, index i and index j

            sampled_pairs = np.random.choice(len(datapoint_outputs), (sample_rotations_pairs_count, 2))
            for pair in sampled_pairs:
                index1 = pair[0]
                index2 = pair[1]

                encoding1 = datapoint_outputs[index1]
                encoding2 = datapoint_outputs[index2]

                rotation_adjacency_loss += torch.norm(encoding1 - encoding2, p=2)

            rotation_adjacency_loss /= sample_rotations_pairs_count
            rotation_adjacency_loss *= scale_rotation_adjacency_loss

            datapoint_mean = torch.mean(torch.stack(datapoint_outputs), dim=0)
            datapoint_means.append(datapoint_mean)

        # ADJACENCY LOSS FOR DATAPOINTS
        # samples datapoints and compares their means representations, and adjusts them to be further / closer to each other

        sampled_connections = storage.sample_non_adjacent_datapoints(sampled_datapoints_count)
        for pair in sampled_connections:
            datapoint1_index = storage.get_datapoint_data_tensor_index_by_name(pair["start"])
            datapoint2_index = storage.get_datapoint_data_tensor_index_by_name(pair["end"])
            datapoint1 = datapoint_means[datapoint1_index]
            datapoint2 = datapoint_means[datapoint2_index]

            distance = torch.norm(datapoint1 - datapoint2, p=2)
            expected_distance = pair["distance"] * DATAPOINTS_DISTANCE_FACTOR

            datapoint_loss += ((distance - expected_distance) / DATAPOINTS_DISTANCE_FACTOR) ** 2

        if sampled_datapoints_count != 0:
            datapoint_loss /= sampled_datapoints_count
        datapoint_loss *= scale_datapoint_loss

        optimizer.step()

        rotation_average_loss += rotation_adjacency_loss.item()
        datapoint_average_loss += datapoint_loss.item()
        epoch_loss += rotation_adjacency_loss + datapoint_loss

        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            rotation_average_loss /= epoch_print_rate
            datapoint_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print(f"EPOCH:{epoch}/{num_epochs}")
            print(f"ROTATION LOSS:{rotation_average_loss} DATASET LOSS:{datapoint_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            rotation_average_loss = 0
            datapoint_average_loss = 0

    return permutor


def run_ai_init():
    permutor = Permutor()
    # permutor = PermutorRawDNN()
    train_permutor(permutor, epochs=2000)


def run_ai():
    train_data = [torch.tensor(datapoint) for datapoint in storage.get_pure_sensor_data()]
    permutor = ImprovedPermutor()
    trained_permutor = train_permutor2(permutor, train_data, epochs=3000)
    return permutor


def evaluate_permutor(permutor: Permutor, storage: Storage) -> None:
    # evaluate difference between permutations ( should be close to 0 )
    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data()))
    norm_sum = 0
    count = 0

    datapoint_means = []
    datapoint_outputs_array = []
    for datapoint in train_data:
        datapoint_outputs = []
        for rotation in datapoint:
            output = permutor(rotation.unsqueeze(0).unsqueeze(0))
            output = output.squeeze(0)
            datapoint_outputs.append(output)

        datapoint_outputs_array.append(datapoint_outputs)

        for i in range(len(datapoint_outputs)):
            for j in range(i + 1, len(datapoint_outputs)):
                count += 1
                norm_sum += torch.norm(datapoint_outputs[i] - datapoint_outputs[j], p=2)

        datapoint_mean = torch.mean(torch.stack(datapoint_outputs), dim=0)
        datapoint_means.append(datapoint_mean)

    print("Permutations distance: ", norm_sum / count)
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
                "sum": 0,
                "count": 0
            }

        distance_between_embeddings = torch.norm(datapoint1 - datapoint2, p=2)
        # if distance == 5:
        #     print(f"Distance between {start_name} and {end_name}: {distance_between_embeddings:.4f}")
        #     print(f"Expected distance: {distance:.4f}")
        #     print(datapoint1)
        #     print(datapoint2)
        #
        #     print("--------------------------------------------------"
        #           )
        #     print(datapoint_outputs_array[datapoint1_index])
        #     print(datapoint_outputs_array[datapoint2_index])
        #     return

        avg_distances[f"{distance}"]["sum"] += distance_between_embeddings
        avg_distances[f"{distance}"]["count"] += 1

    for distance in avg_distances:
        avg_distances[distance]["sum"] /= avg_distances[distance]["count"]
        print(f"Average distance for distance {distance}: {avg_distances[distance]['sum']:.4f}")


def run_tests(autoencoder):
    global storage
    evaluate_permutor(autoencoder, storage)


def run_new_ai() -> None:
    permutor = run_ai()
    save_ai_manually("permutor_other", permutor)
    run_tests(permutor)


def load_ai() -> None:
    global model
    model = load_manually_saved_ai("permutor_other.pth")
    run_tests(model)


def run_permutor() -> None:
    global storage
    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.normalize_all_data_super()

    run_new_ai()
    # load_ai()


storage: StorageSuperset = StorageSuperset()
model = Permutor()
