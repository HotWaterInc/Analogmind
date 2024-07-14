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
    evaluate_adjacency_properties_super
from src.ai.models.permutor import ImprovedPermutor


class AutoencoderPostPermutor(BaseAutoencoderModel):
    def __init__(self):
        super(AutoencoderPostPermutor, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(16, 24),
            nn.LeakyReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(24, 16),
            nn.Tanh(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU()
        )

    def encoder_training(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.encoder1(x)
        encoded = self.encoder2(l1)
        return encoded

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_training(x)

    def decoder_training(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.decoder1(x)
        decoded = self.decoder2(l1)
        # assumes input data from permutor is already normalized between 0 and 1
        decoded = nn.functional.normalize(decoded, p=2, dim=1)
        return decoded

    def decoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(x)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_training(x)
        decoded = self.decoder_training(encoded)
        return decoded

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)


def reconstruction_handling(autoencoder: BaseAutoencoderModel, data: any, criterion: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    enc = autoencoder.encoder_training(data)
    dec = autoencoder.decoder_training(enc)
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
        data_point1 = storage.get_datapoint_data_tensor_by_name_super(pair["start"])
        data_point2 = storage.get_datapoint_data_tensor_by_name_super(pair["end"])
        batch_datapoint1.append(data_point1)
        batch_datapoint2.append(data_point2)

    batch_datapoint1 = torch.stack(batch_datapoint1)
    batch_datapoint2 = torch.stack(batch_datapoint2)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    distance = torch.sum(torch.norm((encoded_i - encoded_j), p=2))

    average_distance += distance.item() / adjacent_sample_size
    adjacent_distance_loss += distance / adjacent_sample_size * scale_adjacent_distance_loss

    return adjacent_distance_loss, average_distance


def non_adjacent_distance_handling(autoencoder: BaseAutoencoderModel, non_adjacent_sample_size: int,
                                   scale_non_adjacent_distance_loss: float, distance_factor: float) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other
    """
    sampled_pairs = storage.sample_datapoints_adjacencies(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []

    for pair in sampled_pairs:
        datapoint1 = storage.get_datapoint_data_tensor_by_name_super(pair["start"])
        datapoint2 = storage.get_datapoint_data_tensor_by_name_super(pair["end"])

        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1)
    batch_datapoint2 = torch.stack(batch_datapoint2)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] * distance_factor for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance)

    non_adjacent_distance_loss = (distance - expected_distance) / distance_factor
    non_adjacent_distance_loss = torch.square(non_adjacent_distance_loss)
    non_adjacent_distance_loss = non_adjacent_distance_loss * scale_non_adjacent_distance_loss
    non_adjacent_distance_loss = torch.mean(non_adjacent_distance_loss)

    return non_adjacent_distance_loss


def train_autoencoder_with_distance_constraint(autoencoder: BaseAutoencoderModel, epochs: int) -> BaseAutoencoderModel:
    """
    Trains the autoencoder with 2 additional losses apart from the reconstruction loss:
    - adjacent distance loss: keeps adjacent pairs close to each other
    - non-adjacent distance loss: keeps non-adjacent pairs far from each other ( in a proportional way to the distance
    between them inferred from the data )
    """

    # PARAMETERS
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    num_epochs = epochs
    scale_reconstruction_loss = 1
    scale_adjacent_distance_loss = 3
    scale_non_adjacent_distance_loss = 0.75

    adjacent_sample_size = 100
    non_adjacent_sample_size = 224

    epoch_average_loss = 0
    reconstruction_average_loss = 0
    adjacent_average_loss = 0
    non_adjacent_average_loss = 0

    epoch_print_rate = 5000
    DISTANCE_CONSTANT = 0.5

    train_data_names = storage.get_sensor_data_names()
    storage.build_permuted_data_raw()
    storage.build_permuted_data_random_rotations()
    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))

    best_loss = 10000000
    stagnation_streak = 0

    for epoch in range(num_epochs):
        if (epoch % 25 == 0):
            storage.build_permuted_data_random_rotations()
            train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))

        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, criterion, scale_reconstruction_loss)
        reconstruction_loss.backward()

        # ADJACENT DISTANCE LOSS
        adjacent_distance_loss = torch.tensor(0.0)
        adjacent_distance_loss, average_distance_adjacent = adjacent_distance_handling(autoencoder,
                                                                                       adjacent_sample_size,
                                                                                       scale_adjacent_distance_loss)
        adjacent_distance_loss.backward()

        # NON-ADJACENT DISTANCE LOSS
        non_adjacent_distance_loss = torch.tensor(0.0)
        non_adjacent_distance_loss = non_adjacent_distance_handling(autoencoder, non_adjacent_sample_size,
                                                                    scale_non_adjacent_distance_loss,
                                                                    distance_factor=DISTANCE_CONSTANT)
        non_adjacent_distance_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + adjacent_distance_loss.item() + non_adjacent_distance_loss.item()

        epoch_average_loss += epoch_loss
        reconstruction_average_loss += reconstruction_loss.item()
        adjacent_average_loss += adjacent_distance_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            adjacent_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate

            if epoch_average_loss < best_loss:
                best_loss = epoch_average_loss
                stagnation_streak = 0

            if epoch_average_loss >= best_loss:
                stagnation_streak += 1

            if stagnation_streak >= 10:
                break

            # Print average loss for this epoch
            print(f"EPOCH:{epoch}/{num_epochs} - streak: {stagnation_streak}")
            # print(f"average distance between adjacent: {average_distance_adjacent}")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            adjacent_average_loss = 0
            non_adjacent_average_loss = 0

    return autoencoder


def run_ai():
    autoencoder = AutoencoderPostPermutor()
    train_autoencoder_with_distance_constraint(autoencoder, epochs=100000)
    return autoencoder


def run_tests(autoencoder):
    global storage

    evaluate_reconstruction_error_super(autoencoder, storage)
    avg_distance_adj = evaluate_distances_between_pairs_super(autoencoder, storage)
    evaluate_adjacency_properties_super(autoencoder, storage, avg_distance_adj)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("autoenc_dynamic10k.pth")
    autoencoder = load_manually_saved_ai("autoencod_perm100k.pth")
    global storage
    storage.build_permuted_data_raw()

    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = run_ai()
    save_ai_manually("autoencod_permutated2", autoencoder)
    run_tests(autoencoder)


def run_permuted_autoencoder2() -> None:
    global storage
    global permutor

    permutor = load_manually_saved_ai("permutor10k.pth")
    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()
    storage.set_permutor(permutor)

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = StorageSuperset2()
permutor: ImprovedPermutor = None
