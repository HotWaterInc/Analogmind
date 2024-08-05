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

import torch
import torch.nn as nn
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, ResidualBlockSmallLayerNorm, AttentionLayer, \
    ResidualBlockSmallBatchNormWithAttention


class PermutorShift(nn.Module):
    def __init__(self, dropout_rate: float = 0.2,
                 hidden_size: int = 1024, num_blocks: int = 2, input_output_size=512,
                 concatenated_instances: int = 6):
        super(PermutorShift, self).__init__()

        self.concatenated_instances = concatenated_instances
        input_output_size *= concatenated_instances
        self.input_layer = nn.Linear(input_output_size, hidden_size)

        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, input_output_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feedforward through the network
        x = self.input_layer(x)
        x = self.leaky_relu(x)

        for block in self.encoding_blocks:
            x = block(x)
        return x


def reconstruction_handling_raw(autoencoder: BaseAutoencoderModel,
                                scale_reconstruction_loss: int = 1) -> torch.Tensor:
    global storage
    sampled_count = 25
    sampled_points = storage.sample_n_random_datapoints(sampled_count)
    input_data = []
    target_data = []
    accumulated_loss = torch.tensor(0.0)

    for point in sampled_points:
        for i in range(OFFSETS_PER_DATAPOINT):
            sampled_full_rotation = array_to_tensor(
                storage.get_point_rotations_with_full_info_random_offset_concatenated(point, ROTATIONS_PER_FULL))
            target_data_north = array_to_tensor(
                storage.get_point_rotations_with_full_info_set_offset_concatenated(point, ROTATIONS_PER_FULL))

            input_data.append(sampled_full_rotation)

    input_data = torch.stack(input_data).to(device=device)
    positional_encoding, thetas_encoding = autoencoder.encoder_training(input_data)
    dec = autoencoder.decoder_training(positional_encoding, thetas_encoding)

    criterion_reconstruction = nn.MSELoss()

    accumulated_loss += criterion_reconstruction(dec, input_data)
    return accumulated_loss * scale_reconstruction_loss


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

    epoch_print_rate = 100

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

        # reconstruction_loss.backward(retain_graph=True)

        # loss_same_position.backward(retain_graph=True)
        # loss_same_rotation.backward(retain_graph=True)

        # ratio_loss_rotation.backward()

        rotation_encoding_change_on_position.backward(retain_graph=True)
        rotation_encoding_change_on_rotation2 = 1 / rotation_encoding_change_on_rotation
        rotation_encoding_change_on_rotation2.backward(retain_graph=True)

        # ratio_loss_rotation.backward(retain_graph=True)

        # position_encoding_change_on_position.backward(retain_graph=True)
        # position_encoding_change_on_rotation.backward(retain_graph=True)
        # ratio_loss_position.backward(retain_graph=True)

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
    evaluation_position_rotation_embeddings(autoencoder, storage)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("abstract_block_v1_0.019.pth")
    autoencoder = load_manually_saved_ai("autoencod_abstract_block2.pth")

    global storage
    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = PermutorShift()
    train_autoencoder_abstraction_block(autoencoder, 5001, True)
    save_ai_manually("autoencod_abstract_block", autoencoder)
    run_tests(autoencoder)


def run_autoencoder_abstraction_block_images() -> None:
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

ROTATIONS_PER_FULL = 6
OFFSETS_PER_DATAPOINT = 23
TOTAL_ROTATIONS = 24
