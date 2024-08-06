import torch
import time
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super, evaluate_reconstruction_error_super_fist_rotation
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display
from .evaluation_abstract_block import evaluation_position_rotation_embeddings, \
    evaluation_position_rotation_embeddings_img1, evaluation_position_rotation_embeddings_img1_on_00

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)


class ResidualBlockSmallLayerNorm(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlockSmallLayerNorm, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlockSmallBatchNormWithAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlockSmallBatchNormWithAttention, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            AttentionLayer(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlockSmallBatchNorm(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlockSmallBatchNorm, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


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


def _make_layer(in_features, out_features):
    layer = nn.Sequential(
        nn.Linear(in_features, out_features),
        # nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )
    return layer


class AutoencoderAbstractionBlockImg1(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, position_embedding_size: int = 96, thetas_embedding_size: int = 96,
                 hidden_size: int = 1024 + 1024, num_blocks: int = 1, input_output_size=512,
                 concatenated_instances: int = 1):
        super(AutoencoderAbstractionBlockImg1, self).__init__()

        self.concatenated_instances = concatenated_instances
        input_output_size *= concatenated_instances
        self.input_layer = nn.Linear(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.positional_encoder = _make_layer(hidden_size, position_embedding_size)
        self.thetas_encoder = _make_layer(hidden_size, thetas_embedding_size)

        self.decoder_initial_layer = _make_layer(position_embedding_size + thetas_embedding_size, hidden_size)
        self.decoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.output_layer = nn.Linear(hidden_size, input_output_size)
        self.leaky_relu = nn.LeakyReLU()
        self.embedding_size = position_embedding_size

    def encoder_training(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Feedforward through the network
        x = self.input_layer(x)
        x = self.leaky_relu(x)

        for block in self.encoding_blocks:
            x = block(x)

        position = self.positional_encoder(x)
        thetas = self.thetas_encoder(x)

        return position, thetas

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        position, _ = self.encoder_training(x)
        return position

    def decoder_training(self, positional_encoding, rotational_encoding) -> torch.Tensor:
        x = torch.cat([positional_encoding, rotational_encoding], dim=-1)
        x = self.decoder_initial_layer(x)
        x = self.leaky_relu(x)

        for block in self.decoding_blocks:
            x = block(x)

        x = self.output_layer(x)

        # Reshape the output back to the original shape
        # batch_size = x.shape[0]
        # x = x.view(batch_size, self.concatenated_instances, -1)

        return x

    def decoder_inference(self, positional_encoding, rotational_encoding) -> torch.Tensor:
        return self.decoder_training(positional_encoding, rotational_encoding)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        positional_encoding, rotational_encoding = self.encoder_training(x)
        return self.decoder_training(positional_encoding, rotational_encoding)

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)

    def get_embedding_size(self) -> int:
        return self.embedding_size


def reconstruction_handling_raw(autoencoder: BaseAutoencoderModel,
                                scale_reconstruction_loss: int = 1) -> torch.Tensor:
    global storage
    sampled_count = 25
    sampled_points = storage.sample_n_random_datapoints(sampled_count)
    data = []
    accumulated_loss = torch.tensor(0.0)

    for point in sampled_points:
        for i in range(OFFSETS_PER_DATAPOINT):
            sampled_full_rotation = array_to_tensor(
                storage.get_point_rotations_with_full_info_random_offset_concatenated(point, ROTATIONS_PER_FULL))
            data.append(sampled_full_rotation)

    data = torch.stack(data).to(device=device)
    positional_encoding, thetas_encoding = autoencoder.encoder_training(data)
    dec = autoencoder.decoder_training(positional_encoding, thetas_encoding)

    criterion_reconstruction = nn.MSELoss()

    accumulated_loss += criterion_reconstruction(dec, data)
    return accumulated_loss * scale_reconstruction_loss


def reconstruction_handling_with_freezing(autoencoder: BaseAutoencoderModel,
                                          scale_reconstruction_loss: float = 1) -> any:
    global storage
    sampled_count = 25
    sampled_points = storage.sample_n_random_datapoints(sampled_count)
    data = []

    loss_reconstruction = torch.tensor(0.0, device=device)
    loss_freezing_same_position = torch.tensor(0.0, device=device)
    loss_freezing_same_rotation = torch.tensor(0.0, device=device)

    for point in sampled_points:
        for i in range(OFFSETS_PER_DATAPOINT):
            sampled_full_rotation = array_to_tensor(
                storage.get_point_rotations_with_full_info_set_offset_concatenated(point, ROTATIONS_PER_FULL, i))
            data.append(sampled_full_rotation)

    data = torch.stack(data).to(device=device)

    positional_encoding, thetas_encoding = autoencoder.encoder_training(data)
    dec = autoencoder.decoder_training(positional_encoding, thetas_encoding)

    criterion_reconstruction = nn.MSELoss()

    loss_reconstruction = criterion_reconstruction(dec, data)
    # traverses each OFFSET_PER_DATAPOINT batch, selects tensors from each batch and calculates the loss

    position_encoding_change_on_position = torch.tensor(0.0, device=device)
    position_encoding_change_on_rotation = torch.tensor(0.0, device=device)

    rotation_encoding_change_on_position = torch.tensor(0.0, device=device)
    rotation_encoding_change_on_rotation = torch.tensor(0.0, device=device)

    # rotation changes, position stays the same
    for i in range(sampled_count):
        start_index = i * OFFSETS_PER_DATAPOINT
        end_index = (i + 1) * OFFSETS_PER_DATAPOINT
        positional_encs = positional_encoding[start_index:end_index]
        rotational_encs = thetas_encoding[start_index:end_index]

        loss_freezing_same_position += torch.cdist(positional_encs, positional_encs).mean()

        position_encoding_change_on_rotation += torch.cdist(positional_encs, positional_encs).mean()
        rotation_encoding_change_on_rotation += torch.cdist(rotational_encs, rotational_encs).mean()

    loss_freezing_same_position /= sampled_count

    position_encoding_change_on_rotation /= sampled_count
    rotation_encoding_change_on_rotation /= sampled_count

    rotation_constant_array_rotation_embeddings = []
    rotation_constant_array_position_embeddings = []

    # putting same rotations in a list one after another
    for rotation_offset in range(OFFSETS_PER_DATAPOINT):
        for position_index in range(sampled_count):
            idx = position_index * OFFSETS_PER_DATAPOINT + rotation_offset
            rotation_constant_array_rotation_embeddings.append(thetas_encoding[idx])
            rotation_constant_array_position_embeddings.append(positional_encoding[idx])

    rotation_constant_array_rotation_embeddings = torch.stack(rotation_constant_array_rotation_embeddings).to(
        device=device)
    rotation_constant_array_position_embeddings = torch.stack(rotation_constant_array_position_embeddings).to(
        device=device
    )

    # position changes, rotation stays the same
    for i in range(OFFSETS_PER_DATAPOINT):
        start_index = i * sampled_count
        end_index = (i + 1) * sampled_count

        rotational_encs = rotation_constant_array_rotation_embeddings[start_index:end_index]
        positional_encs = rotation_constant_array_position_embeddings[start_index:end_index]

        loss_freezing_same_rotation += torch.cdist(rotational_encs, rotational_encs).mean()

        position_encoding_change_on_position += torch.cdist(positional_encs, positional_encs).mean()
        rotation_encoding_change_on_position += torch.cdist(rotational_encs, rotational_encs).mean()

    loss_freezing_same_rotation /= sampled_count

    position_encoding_change_on_position /= sampled_count
    rotation_encoding_change_on_position /= sampled_count

    loss_reconstruction *= scale_reconstruction_loss
    loss_freezing_same_position *= 1
    loss_freezing_same_rotation *= 1

    ratio_loss_position = position_encoding_change_on_rotation / position_encoding_change_on_position
    ratio_loss_rotation = rotation_encoding_change_on_position / rotation_encoding_change_on_rotation

    return {
        "loss_reconstruction": loss_reconstruction,
        "loss_freezing_same_position": loss_freezing_same_position,
        "loss_freezing_same_rotation": loss_freezing_same_rotation,
        "ratio_loss_position": ratio_loss_position,
        "ratio_loss_rotation": ratio_loss_rotation,
        "rotation_encoding_change_on_position": rotation_encoding_change_on_position,
        "rotation_encoding_change_on_rotation": rotation_encoding_change_on_rotation,
        "position_encoding_change_on_position": position_encoding_change_on_position,
        "position_encoding_change_on_rotation": position_encoding_change_on_rotation
    }


def train_autoencoder_abstraction_block(autoencoder: BaseAutoencoderModel, epochs: int,
                                        pretty_print: bool = False) -> nn.Module:
    # PARAMETERS
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001, amsgrad=True)
    autoencoder = autoencoder.to(device=device)

    num_epochs = epochs

    scale_reconstruction_loss = 1
    scale_freezing_loss = 1

    epoch_average_loss = 0

    loss_same_position_average_loss = 0
    loss_same_rotation_average_loss = 0
    loss_ratio_position_average_loss = 0
    loss_ratio_rotation_average_loss = 0

    reconstruction_average_loss = 0

    epoch_print_rate = 1000

    SHUFFLE_RATE = 5

    if pretty_print:
        set_pretty_display(epoch_print_rate, "Epoch batch")
        pretty_display_start(0)

    rotation_encoding_change_on_position_avg = 0
    rotation_encoding_change_on_rotation_avg = 0

    position_encoding_change_on_position_avg = 0
    position_encoding_change_on_rotation_avg = 0

    for epoch in range(num_epochs):

        reconstruction_loss = torch.tensor(0.0)
        loss_same_position = torch.tensor(0.0)
        loss_same_rotation = torch.tensor(0.0)
        ratio_loss_position = torch.tensor(0.0)
        ratio_loss_rotation = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        losses_json = reconstruction_handling_with_freezing(
            autoencoder,
            scale_reconstruction_loss)

        reconstruction_loss = losses_json["loss_reconstruction"]
        ratio_loss_position = losses_json["ratio_loss_position"]
        ratio_loss_rotation = losses_json["ratio_loss_rotation"]
        loss_same_position = losses_json["loss_freezing_same_position"]
        loss_same_rotation = losses_json["loss_freezing_same_rotation"]

        rotation_encoding_change_on_position = losses_json["rotation_encoding_change_on_position"]
        rotation_encoding_change_on_rotation = losses_json["rotation_encoding_change_on_rotation"]

        position_encoding_change_on_position = losses_json["position_encoding_change_on_position"]
        position_encoding_change_on_rotation = losses_json["position_encoding_change_on_rotation"]

        reconstruction_loss.backward(retain_graph=True)
        loss_same_position.backward(retain_graph=True)
        loss_same_rotation.backward(retain_graph=True)
        ratio_loss_rotation.backward(retain_graph=True)
        ratio_loss_position.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + loss_same_position.item() + loss_same_rotation.item() + ratio_loss_position.item() + ratio_loss_rotation.item()
        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        loss_same_position_average_loss += loss_same_position.item()
        loss_same_rotation_average_loss += loss_same_rotation.item()
        loss_ratio_position_average_loss += ratio_loss_position.item()
        loss_ratio_rotation_average_loss += ratio_loss_rotation.item()

        rotation_encoding_change_on_position_avg += rotation_encoding_change_on_position.item()
        rotation_encoding_change_on_rotation_avg += rotation_encoding_change_on_rotation.item()

        position_encoding_change_on_position_avg += position_encoding_change_on_position.item()
        position_encoding_change_on_rotation_avg += position_encoding_change_on_rotation.item()

        if pretty_print:
            pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:

            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            loss_same_position_average_loss /= epoch_print_rate
            loss_same_rotation_average_loss /= epoch_print_rate
            loss_ratio_position_average_loss /= epoch_print_rate
            loss_ratio_rotation_average_loss /= epoch_print_rate

            rotation_encoding_change_on_position_avg /= epoch_print_rate
            rotation_encoding_change_on_rotation_avg /= epoch_print_rate

            position_encoding_change_on_position_avg /= epoch_print_rate
            position_encoding_change_on_rotation_avg /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs} ")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | LOSS SAME POSITION:{loss_same_position_average_loss} | LOSS SAME ROTATION:{loss_same_rotation_average_loss} | RATIO LOSS POSITION:{loss_ratio_position_average_loss} | RATIO LOSS ROTATION:{loss_ratio_rotation_average_loss}")
            print(
                f"Changes of rotation encoding on position: {rotation_encoding_change_on_position_avg} | Changes of rotation encoding on rotation: {rotation_encoding_change_on_rotation_avg}")
            print(
                f"Changes of position encoding on position: {position_encoding_change_on_position_avg} | Changes of position encoding on rotation: {position_encoding_change_on_rotation_avg}")

            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            loss_same_position_average_loss = 0
            loss_same_rotation_average_loss = 0
            loss_ratio_position_average_loss = 0
            loss_ratio_rotation_average_loss = 0

            if pretty_print:
                pretty_display_reset()
                pretty_display_start(epoch)

    return autoencoder


def run_tests(autoencoder):
    global storage
    # evaluation_position_rotation_embeddings_img1(autoencoder, storage)
    evaluation_position_rotation_embeddings_img1_on_00(autoencoder, storage)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("abstract_block_v1_0.019.pth")
    autoencoder = load_manually_saved_ai("abstraction_block_1img_saved.pth")

    global storage
    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = AutoencoderAbstractionBlockImg1()
    autoencoder = train_autoencoder_abstraction_block(autoencoder, 15001, True)
    save_ai_manually("abstract_block_img1", autoencoder)
    run_tests(autoencoder)


def run_abstraction_block_images_img1() -> None:
    global storage

    grid_data = 5

    storage.load_raw_data_from_others(f"data{grid_data}x{grid_data}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_data}x{grid_data}_connections.json")

    # selects first rotation
    storage.build_permuted_data_random_rotations_rotation0()

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = StorageSuperset2()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROTATIONS_PER_FULL = 1
OFFSETS_PER_DATAPOINT = 24
TOTAL_ROTATIONS = 24
