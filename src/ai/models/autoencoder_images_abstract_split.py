import torch
import time
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


class AutoencoderAbstractionBlock(nn.Module):
    def __init__(self, drop_rate: float = 0.2):
        super(AutoencoderAbstractionBlock, self).__init__()
        self.embedding_size = 128

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate=0, final_layer=True),
        ])

        self.positional_encoder = nn.ModuleList([
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, self.embedding_size, drop_rate=0, final_layer=True)
        ])

        self.rotational_encoder = nn.ModuleList([
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, self.embedding_size, drop_rate=0, final_layer=True),
        ])

        # Decoder layers
        self.positional_decoder = nn.ModuleList([
            self._make_layer(self.embedding_size, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate=0),
        ])

        self.rotational_decoder = nn.ModuleList([
            self._make_layer(self.embedding_size, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate=0),
        ])

        self.decoder_layers = nn.ModuleList([
            self._make_layer(1024, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate),
            self._make_layer(512, 512, drop_rate=0, final_layer=True),
        ])

    def _make_layer(self, in_features, out_features, drop_rate, final_layer=False):
        layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(),
        )
        if not final_layer:
            layer.add_module('dropout', nn.Dropout(drop_rate))
        return layer

    def encoder_training(self, x: torch.Tensor) -> any:
        initial_data = x
        for layer in self.encoder_layers:
            x = layer(x)

        # x += initial_data
        positional_encoder = x
        rotational_encoder = x

        for layer in self.positional_encoder:
            positional_encoder = layer(positional_encoder)

        for layer in self.rotational_encoder:
            rotational_encoder = layer(rotational_encoder)

        return positional_encoder, rotational_encoder

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_training(x)

    def decoder_training(self, x: torch.Tensor, positional_encoding, rotational_encoding) -> torch.Tensor:
        for layer in self.positional_decoder:
            positional_encoding = layer(positional_encoding)

        for layer in self.rotational_decoder:
            rotational_encoding = layer(rotational_encoding)

        x = torch.cat((positional_encoding, rotational_encoding), dim=1)
        for layer in self.decoder_layers:
            x = layer(x)

        return x

    def decoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        positional_encoding, rotational_encoding = self.encoder_training(x)
        return self.decoder_training(x, positional_encoding, rotational_encoding)

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)

    def get_embedding_size(self) -> int:
        return self.embedding_size


def reconstruction_handling(autoencoder: BaseAutoencoderModel, data: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    dec = autoencoder.forward_training(data)
    criterion = nn.MSELoss()

    return criterion(dec, data) * scale_reconstruction_loss


def adjacent_distance_handling(autoencoder: BaseAutoencoderModel, adjacent_sample_size: int,
                               scale_adjacent_distance_loss: float) -> tuple[torch.Tensor, float]:
    """
    Keeps adjacent pairs close to each other
    """
    sampled_pairs = storage.sample_adjacent_datapoints_connections(adjacent_sample_size)

    adjacent_distance_loss = torch.tensor(0.0)
    average_distance = 0
    batch_datapoint1 = []
    batch_datapoint2 = []
    for pair in sampled_pairs:
        # keep adjacent close to each other
        data_point1 = storage.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        data_point2 = storage.get_datapoint_data_tensor_by_name_permuted(pair["end"])
        batch_datapoint1.append(data_point1)
        batch_datapoint2.append(data_point2)

    batch_datapoint1 = torch.stack(batch_datapoint1)
    batch_datapoint2 = torch.stack(batch_datapoint2)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    # embedding_size = encoded_i.shape[1]

    distance = torch.sum(torch.norm((encoded_i - encoded_j), p=2))

    average_distance += distance.item() / adjacent_sample_size
    adjacent_distance_loss += distance / adjacent_sample_size * scale_adjacent_distance_loss

    return adjacent_distance_loss, average_distance


def non_adjacent_distance_handling(autoencoder: BaseAutoencoderModel, non_adjacent_sample_size: int,
                                   scale_non_adjacent_distance_loss: float, distance_per_neuron: float) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other
    """
    sampled_pairs = storage.sample_datapoints_adjacencies(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []

    for pair in sampled_pairs:
        datapoint1 = storage.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        datapoint2 = storage.get_datapoint_data_tensor_by_name_permuted(pair["end"])

        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1)
    batch_datapoint2 = torch.stack(batch_datapoint2)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] * distance_per_neuron * embedding_size for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance)

    criterion = nn.MSELoss()
    non_adjacent_distance_loss = criterion(distance, expected_distance) * scale_non_adjacent_distance_loss
    return non_adjacent_distance_loss


def permutation_adjustion_handling(autoencoder: BaseAutoencoderModel, samples: int,
                                   scale_permutation_adjustion_loss: float) -> torch.Tensor:
    """
    Keeps the permutation of the data points close to each other
    """
    global storage

    datapoint: List[str] = storage.sample_n_random_datapoints(samples)
    datapoints_data = [storage.get_datapoint_data_tensor_by_name(name) for name in datapoint]
    accumulated_loss = torch.tensor(0.0)
    for datapoint_data in datapoints_data:
        enc = autoencoder.encoder_training(datapoint_data)
        loss = torch.cdist(enc, enc, p=2).mean()
        accumulated_loss += loss

    return accumulated_loss / samples * scale_permutation_adjustion_loss


def train_autoencoder_abstraction_block(autoencoder: nn.Module, epochs: int,
                                        pretty_print: bool = False) -> nn.Module:
    # PARAMETERS
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0015)

    num_epochs = epochs

    scale_reconstruction_loss = 1
    scale_adjacent_distance_loss = 0.1
    scale_non_adjacent_distance_loss = 0.5
    scale_permutation_adjustion_loss = 0.75

    adjacent_sample_size = 100
    non_adjacent_sample_size = 2000
    permutation_sample_size = 100

    epoch_average_loss = 0
    reconstruction_average_loss = 0
    adjacent_average_loss = 0
    non_adjacent_average_loss = 0
    permutation_average_loss = 0

    epoch_print_rate = 250
    DISTANCE_CONSTANT_PER_NEURON = 0.005

    # train_data = array_to_tensor(np.array(storage.get_pure_sensor_data())[0])
    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))

    best_loss = 10000000
    stagnation_streak = 0

    SHUFFLE_RATE = 5

    if pretty_print:
        set_pretty_display(epoch_print_rate, "Epoch batch")
        pretty_display_start(0)

    for epoch in range(num_epochs):
        # if (epoch % SHUFFLE_RATE == 0):
        #     storage.build_permuted_data_random_rotations()
        #     train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))

        reconstruction_loss = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, scale_reconstruction_loss)
        reconstruction_loss.backward()

        # ADJACENT DISTANCE LOSS
        adjacent_distance_loss = torch.tensor(0.0)
        # adjacent_distance_loss, average_distance_adjacent = adjacent_distance_handling(autoencoder,
        #                                                                                adjacent_sample_size,
        #                                                                                scale_adjacent_distance_loss)
        # adjacent_distance_loss.backward()

        # NON-ADJACENT DISTANCE LOSS
        non_adjacent_distance_loss = torch.tensor(0.0)
        # non_adjacent_distance_loss = non_adjacent_distance_handling(autoencoder, non_adjacent_sample_size,
        #                                                             scale_non_adjacent_distance_loss,
        #                                                             distance_per_neuron=DISTANCE_CONSTANT_PER_NEURON)
        # non_adjacent_distance_loss.backward()
        # PERMUTATION ADJUSTION LOSS
        permutation_adjustion_loss = torch.tensor(0.0)
        # permutation_adjustion_loss = permutation_adjustion_handling(autoencoder, permutation_sample_size,
        #                                                             scale_permutation_adjustion_loss)
        # permutation_adjustion_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + adjacent_distance_loss.item() + non_adjacent_distance_loss.item() + permutation_adjustion_loss.item()

        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        adjacent_average_loss += adjacent_distance_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()
        permutation_average_loss += permutation_adjustion_loss.item()

        if pretty_print:
            pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:

            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            adjacent_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate
            permutation_average_loss /= epoch_print_rate

            if epoch_average_loss < best_loss:
                best_loss = epoch_average_loss
                stagnation_streak = 0

            if epoch_average_loss >= best_loss:
                stagnation_streak += 1

            if stagnation_streak >= 10:
                break

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs} - streak: {stagnation_streak}")
            # print(f"average distance between adjacent: {average_distance_adjacent}")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss} | PERMUTATION LOSS:{permutation_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            adjacent_average_loss = 0
            non_adjacent_average_loss = 0
            permutation_average_loss = 0

            if pretty_print:
                pretty_display_reset()
                pretty_display_start(epoch)

    return autoencoder


def run_ai():
    global storage
    autoencoder = AutoencoderAbstractionBlock()
    train_autoencoder_abstraction_block(autoencoder, epochs=4000, pretty_print=True)
    return autoencoder


def run_tests(autoencoder):
    global storage

    evaluate_reconstruction_error_super_fist_rotation(autoencoder, storage)
    # evaluate_reconstruction_error_super(autoencoder, storage, rotations0=False)
    # avg_distance_adj = evaluate_distances_between_pairs_super(autoencoder, storage, rotations0=True)
    # evaluate_adjacency_properties_super(autoencoder, storage, avg_distance_adj, rotation0=True)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("autoenc_dynamic10k.pth")
    autoencoder = load_manually_saved_ai("autoencod_abstract_block.pth")
    global storage
    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = run_ai()
    save_ai_manually("autoencod_abstract_block", autoencoder)
    run_tests(autoencoder)


def run_autoencoder_abstraction_block_images() -> None:
    global storage
    global permutor

    storage.load_raw_data_from_others("data15x15_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others("data15x15_connections.json")
    # selects first rotation
    storage.build_permuted_data_random_rotations_rotation0()

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = StorageSuperset2()
permutor = None
