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


import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedPermutorAttention(nn.Module):
    def __init__(self, input_size=24, embedding_size=64, num_heads=8):
        super(ImprovedPermutorAttention, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.fc_input_size = 64 * (input_size + 6)
        self.fc_encode = nn.Linear(self.fc_input_size, embedding_size)

        # Encoder Attention
        self.encoder_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads)

        # Post-attention linear layers (encoder)
        self.post_attention_fc1 = nn.Linear(embedding_size, embedding_size)
        self.post_attention_fc2 = nn.Linear(embedding_size, embedding_size)

        # Theta prediction
        self.theta_fc = nn.Linear(embedding_size, 1)

        # Decoder Attention
        self.decoder_attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads)

        # Post-attention linear layers (decoder)
        self.post_attention_dec_fc1 = nn.Linear(embedding_size, embedding_size)
        self.post_attention_dec_fc2 = nn.Linear(embedding_size, embedding_size)

        # Reconstruction network
        self.fc_decode = nn.Linear(embedding_size, self.fc_input_size)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2)
        self.deconv1 = nn.ConvTranspose1d(16, 1, kernel_size=7, padding=3)

        self.activation = nn.LeakyReLU()

    def forward_training(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Encoding
        x = torch.cat([x[:, :, -3:], x, x[:, :, :3]], dim=2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        x_flat = x.view(x.size(0), -1)
        embedding = self.fc_encode(x_flat)

        # Apply encoder self-attention
        embedding = embedding.unsqueeze(0)  # Add sequence dimension
        embedding, _ = self.encoder_attention(embedding, embedding, embedding)
        embedding = embedding.squeeze(0)  # Remove sequence dimension

        # Post-attention linear layers (encoder)
        embedding = self.activation(self.post_attention_fc1(embedding))
        embedding = self.activation(self.post_attention_fc2(embedding))

        embedding = F.normalize(embedding, p=2, dim=1)

        theta = torch.sigmoid(self.theta_fc(embedding))

        return embedding, theta

    def forward(self, x):
        embedding, theta = self.forward_training(x)
        return embedding

    def reconstruct(self, embedding, theta):
        # Apply decoder self-attention
        x = embedding.unsqueeze(0)  # Add sequence dimension
        x, _ = self.decoder_attention(x, x, x)
        x = x.squeeze(0)  # Remove sequence dimension

        # Post-attention linear layers (decoder)
        x = self.activation(self.post_attention_dec_fc1(x))
        x = self.activation(self.post_attention_dec_fc2(x))

        # Decoding
        x = self.fc_decode(x)
        x = x.view(-1, 64, self.fc_input_size // 64)

        x = self.activation(self.deconv3(x))
        x = self.activation(self.deconv2(x))
        x = self.deconv1(x)

        # Remove padding
        x = x[:, :, 3:-3]

        return x.squeeze(1)


class ImprovedPermutorAttentionV2(nn.Module):
    def __init__(self, input_size=24, embedding_size=64, num_heads=8):
        super(ImprovedPermutorAttentionV2, self).__init__()

        self.initial_conv = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.fc_input_size = 16 * (input_size + 6)
        self.initial_fc = nn.Linear(self.fc_input_size, embedding_size)

        self.layers = nn.ModuleList()
        for _ in range(4):
            self.layers.append(nn.ModuleDict({
                'attention': nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads),
                'cnn': nn.Conv1d(1, 1, kernel_size=3, padding=1),
                'linear': nn.Linear(embedding_size, embedding_size)
            }))

        self.final_fc = nn.Linear(embedding_size, embedding_size)
        self.theta_fc = nn.Linear(embedding_size, 1)
        self.activation = nn.LeakyReLU()

    def forward_train(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Initial encoding
        x = torch.cat([x[:, :, -3:], x, x[:, :, :3]], dim=2)
        x = self.activation(self.initial_conv(x))

        x_flat = x.view(x.size(0), -1)
        x = self.initial_fc(x_flat)

        # Multiple layers of attention, CNN, and linear transformation
        for i, layer in enumerate(self.layers):
            # Attention
            x_att = x.unsqueeze(0)
            x_att, _ = layer['attention'](x_att, x_att, x_att)
            x_att = x_att.squeeze(0)

            # CNN
            x_cnn = x_att.unsqueeze(1)  # Add channel dimension
            x_cnn = layer['cnn'](x_cnn)
            x_cnn = x_cnn.squeeze(1)  # Remove channel dimension

            # Linear
            x_linear = layer['linear'](x_cnn)

            # Combine and activate
            x = self.activation(x_att + x_cnn + x_linear)

        # Final processing
        x = self.final_fc(x)
        x = F.normalize(x, p=2, dim=1)

        theta = torch.sigmoid(self.theta_fc(x))

        return x, theta

    def forward(self, x):
        x, theta = self.forward_train(x)
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
        anchor_data = storage.get_datapoint_data_tensor_by_name(anchor_name).to(device)
        positive_data = storage.get_datapoint_data_tensor_by_name(deg1_name).to(device)
        negative_data = storage.get_datapoint_data_tensor_by_name(deg2_name).to(device)

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
        datapoint = storage.get_datapoint_data_tensor_by_name(datapoint_name).to(device)
        # Randomly select two rotations of the same datapoint
        anchor, positive = random.sample(list(datapoint), 2)
        # gets one adjacent datapoint
        adjacent_datapoint_names = storage.sample_adjacent_datapoint_at_degree_most(datapoint_name, 5, 4)
        adjacent_datapoints = [storage.get_datapoint_data_tensor_by_name(adjacent_datapoint_name).to(device) for
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


def rotations_distance_loss(permutor, batch_size) -> torch.Tensor:
    global storage
    datapoints_names: List[str] = storage.sample_n_random_datapoints(batch_size)
    datapoints_data: List[torch.Tensor] = [storage.get_datapoint_data_tensor_by_name(datapoint_name).to(device) for
                                           datapoint_name in datapoints_names]
    datapoints_data = [datapoint_data.unsqueeze(1) for datapoint_data in datapoints_data]
    accumulated_loss = torch.tensor(0.0, device=device)  # Create tensor directly on the device

    for datapoint_data in datapoints_data:
        datapoint_data = datapoint_data.to(device)  # Assign the result back to datapoint_data
        outputs: torch.Tensor = permutor(datapoint_data)
        pairwise_distances = torch.cdist(outputs, outputs, p=2)
        accumulated_loss += torch.sum(pairwise_distances)

    accumulated_loss /= batch_size
    return accumulated_loss


def datapoint_distance_loss(permutor, non_adjacent_sample_size: int, distance_factor: float) -> torch.Tensor:
    sampled_pairs = storage.sample_datapoints_adjacencies(non_adjacent_sample_size)
    batch_datapoint1 = []
    batch_datapoint2 = []

    for pair in sampled_pairs:
        datapoint1 = storage.get_datapoint_data_random_rotation_tensor_by_name(pair["start"]).to(device)
        datapoint2 = storage.get_datapoint_data_random_rotation_tensor_by_name(pair["end"]).to(device)
        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1).unsqueeze(1)
    batch_datapoint2 = torch.stack(batch_datapoint2).unsqueeze(1)
    batch_datapoint1 = batch_datapoint1.to(device)
    batch_datapoint2 = batch_datapoint2.to(device)

    encoded_i = permutor(batch_datapoint1)
    encoded_j = permutor(batch_datapoint2)

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)

    expected_distance = [pair["distance"] * distance_factor for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance, device=device)  # Move to device

    datapoint_distances_loss = (distance - expected_distance) / distance_factor
    datapoint_distances_loss = torch.square(datapoint_distances_loss)
    datapoint_distances_loss = torch.mean(datapoint_distances_loss)
    return datapoint_distances_loss.to(device)  # Ensure the final result is on the device


def train_permutor3(permutor, epochs):
    global storage

    optimizer = optim.Adam(permutor.parameters(), lr=0.001)
    BATCH_SIZE = 64

    DISTANCE_THRESHOLD = 0.4

    scale_datapoint_loss = 1
    scale_rotation_loss = 0.1

    datapoints_adjacent_sample = 100

    epoch_print_rate = 1000

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
        datapoint_loss = datapoint_distance_loss(permutor, datapoints_adjacent_sample,
                                                 DISTANCE_THRESHOLD) * scale_datapoint_loss
        datapoint_loss.backward()

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


def train_permutor_attention(permutor, epochs):
    global storage

    optimizer = optim.Adam(permutor.parameters(), lr=0.001)
    BATCH_SIZE = 64

    DISTANCE_THRESHOLD = 0.4

    scale_datapoint_loss = 0.5
    scale_rotation_loss = 0.02

    datapoints_adjacent_sample = 100

    epoch_print_rate = epochs / 20

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
        datapoint_loss = datapoint_distance_loss(permutor, datapoints_adjacent_sample,
                                                 DISTANCE_THRESHOLD) * scale_datapoint_loss
        datapoint_loss.backward()

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

    permutor = ImprovedPermutorAttentionV2().to(device)
    trained_permutor = train_permutor_attention(permutor, epochs=500)
    return trained_permutor


def evaluate_permutor(permutor: ImprovedPermutor, storage: Storage) -> None:
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

    run_new_ai()
    # load_ai()


storage: StorageSuperset = StorageSuperset()
