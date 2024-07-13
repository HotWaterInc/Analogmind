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
        # self.sigmoid = nn.Sigmoid()

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


def adjacent_permutor_loss(permutor, batch_names, criterion) -> torch.Tensor:
    global storage
    anchors = []
    positives = []
    negatives = []
    for j in range(len(batch_names)):
        anchor_name = batch_names[j]
        deg1_name, deg2_name = storage.sample_triplet_anchor_positive_negative(anchor_name)

        # Randomly select a rotation from a different datapoint
        anchor_data = storage.get_datapoint_data_tensor_by_name(anchor_name)
        positive_data = storage.get_datapoint_data_tensor_by_name(deg1_name)
        negative_data = storage.get_datapoint_data_tensor_by_name(deg2_name)

        anchor = random.choice(list(anchor_data))
        positive = random.choice(list(positive_data))
        negative = random.choice(list(negative_data))

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
    return loss


def datapoint_rotation_permutor_loss(permutor, batch_names, criterion) -> torch.Tensor:
    anchors_arr = []
    positives_arr = []
    negatives_arr = []
    for j in range(len(batch_names)):
        datapoint_name = batch_names[j]
        datapoint = storage.get_datapoint_data_tensor_by_name(datapoint_name)
        # Randomly select two rotations of the same datapoint
        anchor, positive = random.sample(list(datapoint), 2)
        # gets one adjacent datapoint
        adjacent_datapoint_names = storage.sample_adjacent_datapoint_at_degree_most(datapoint_name, 5, 4)
        adjacent_datapoints = [storage.get_datapoint_data_tensor_by_name(adjacent_datapoint_name) for
                               adjacent_datapoint_name in adjacent_datapoint_names]
        # Randomly select a rotation from a different datapoint
        negative = random.choice(
            [random.choice(list(adjacent_datapoint)) for adjacent_datapoint in adjacent_datapoints])
        anchors_arr.append(anchor)
        positives_arr.append(positive)
        negatives_arr.append(negative)

    anchors = torch.stack(anchors_arr).unsqueeze(1)
    positives = torch.stack(positives_arr).unsqueeze(1)
    negatives = torch.stack(negatives_arr).unsqueeze(1)

    anchor_out = permutor(anchors)
    positive_out = permutor(positives)
    negative_out = permutor(negatives)

    loss = criterion(anchor_out, positive_out, negative_out)
    return loss


def train_permutor2(permutor, epochs=2000, batch_size=32):
    global storage

    optimizer = optim.Adam(permutor.parameters(), lr=0.001)
    DISTANCE_THRESHOLD = 0.5

    # d and r stand for datapoint and rotation
    # used to differentiate rotations from each other
    # pick d1r1, d1r2, d2r1 ( should have one distance between them )
    criterion1 = nn.TripletMarginLoss(margin=DISTANCE_THRESHOLD)

    # used to differentiate datapoints from each other
    # pick d1r1, d2r1, d3r1 ( should have 2 distance between them)
    criterion2 = nn.TripletMarginLoss(margin=DISTANCE_THRESHOLD * 2)
    epoch_print_rate = 1000

    train_data_names = storage.get_sensor_data_names()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_data_names), batch_size):
            batch_names = train_data_names[i:i + batch_size]

            loss1 = datapoint_rotation_permutor_loss(permutor, batch_names, criterion1)
            # loss2 = adjacent_permutor_loss(permutor, batch_names, criterion2) * 0.2

            optimizer.zero_grad()
            loss1.backward()
            # loss2.backward()
            optimizer.step()

            total_loss += loss1.item()

        if epoch % epoch_print_rate == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    return permutor


def reconstruction_handling(autoencoder: ImprovedPermutor, data: any, criterion: any,
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


def run_ai():
    permutor = ImprovedPermutor()
    trained_permutor = train_permutor2(permutor, epochs=10000)
    return trained_permutor


def evaluate_permutor(permutor: ImprovedPermutor, storage: Storage) -> None:
    # evaluate difference between permutations ( should be close to 0 )
    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data()))
    norm_sum = 0
    count = 0
    permutation_distance_array = []

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

        randi = random.randint(0, len(datapoint_outputs) - 1)
        randj = random.randint(0, len(datapoint_outputs) - 1)

        permutation_distance_array.append(torch.norm(datapoint_outputs[randi] - datapoint_outputs[randj], p=2).item())

        datapoint_mean = torch.mean(torch.stack(datapoint_outputs), dim=0)
        datapoint_means.append(datapoint_mean)

    print("Permutations distance: ", norm_sum / count)
    # print("Permutations distance array: ", sorted(permutation_distance_array))
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
    save_ai_manually("permutor_new", permutor)
    run_tests(permutor)


def load_ai() -> None:
    global model
    model = load_manually_saved_ai("permutor_final1.pth")
    run_tests(model)


def run_permutor() -> None:
    global storage
    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()

    # run_new_ai()
    load_ai()


storage: StorageSuperset = StorageSuperset()
model = ImprovedPermutor()
