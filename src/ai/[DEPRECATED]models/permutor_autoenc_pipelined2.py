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


class AutoencoderPostPermutor(BaseAutoencoderModel):
    def __init__(self, drop_rate: float = 0.2):
        super(AutoencoderPostPermutor, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(24, 48),
            nn.LeakyReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Dropout(drop_rate),
        )

        self.encoder_final = nn.Sequential(
            nn.Linear(48, 24),
            nn.LeakyReLU(),
        )
        self.encoder_final_leakeage = nn.Sequential(
            nn.Linear(48, 8),
            nn.Tanh(),
        )

        self.decoder_init = nn.Sequential(
            nn.Linear(32, 48),
            nn.LeakyReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(48, 48),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(48, 24),
            nn.Tanh()
        )

    def encoder_training(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.encoder1(x)
        l2 = self.encoder2(l1)
        encoded = self.encoder_final(l2)
        leakeage = self.encoder_final_leakeage(l2)
        return encoded

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_training(x)

    def decoder_training(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.decoder_init(x)
        l2 = self.decoder1(l1)
        decoded = self.decoder2(l2)
        return decoded

    def decoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(x)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.encoder2(x)
        encoded = self.encoder_final(x)
        leakeage = self.encoder_final_leakeage(x)

        x = self.decoder_init(torch.cat([encoded, leakeage], dim=1))
        x = self.decoder1(x)
        decoded = self.decoder2(x)
        return decoded

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)


def reconstruction_handling(autoencoder: BaseAutoencoderModel, data: any, criterion: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    dec = autoencoder.forward_training(data)
    return criterion(dec, data) * scale_reconstruction_loss


def adjacent_distance_handling(autoencoder: BaseAutoencoderModel, adjacent_sample_size: int,
                               scale_adjacent_distance_loss: float) -> tuple[torch.Tensor, float]:
    """
    Keeps adjacent pairs close to each other
    """
    sampled_pairs = storage_raw.sample_adjacent_datapoints_connections(adjacent_sample_size)

    adjacent_distance_loss = torch.tensor(0.0)
    average_distance = 0
    batch_datapoint1 = []
    batch_datapoint2 = []
    for pair in sampled_pairs:
        # keep adjacent close to each other
        data_point1 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        data_point2 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["end"])
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


count = 0


def non_adjacent_distance_handling(autoencoder: BaseAutoencoderModel, non_adjacent_sample_size: int,
                                   scale_non_adjacent_distance_loss: float, distance_factor: float) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other
    """
    global count
    sampled_pairs = storage_raw.sample_datapoints_adjacencies_cheated(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []

    for pair in sampled_pairs:
        datapoint1 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        datapoint2 = storage_raw.get_datapoint_data_tensor_by_name_permuted(pair["end"])

        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1)
    batch_datapoint2 = torch.stack(batch_datapoint2)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] * distance_factor for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance)
    count += 1

    criterion = nn.L1Loss()
    non_adjacent_distance_loss = criterion(distance, expected_distance) * scale_non_adjacent_distance_loss
    return non_adjacent_distance_loss


def permutation_adjustion_handling(autoencoder: BaseAutoencoderModel, samples: int,
                                   scale_permutation_adjustion_loss: float) -> torch.Tensor:
    """
    Keeps the permutation of the data points close to each other
    """
    datapoint: List[str] = storage_raw.sample_n_random_datapoints(samples)
    datapoints_data = [storage_raw.get_datapoint_data_tensor_by_name(name) for name in datapoint]
    accumulated_loss = torch.tensor(0.0)
    for datapoint_data in datapoints_data:
        enc = autoencoder.encoder_training(datapoint_data)
        loss = torch.cdist(enc, enc, p=2).mean()
        accumulated_loss += loss

    return accumulated_loss / samples * scale_permutation_adjustion_loss


def train_autoencoder_with_distance_constraint(autoencoder: BaseAutoencoderModel, epochs: int) -> BaseAutoencoderModel:
    """
    Trains the autoencoder with 2 additional losses apart from the reconstruction loss:
    - adjacent distance loss: keeps adjacent pairs close to each other
    - non-adjacent distance loss: keeps non-adjacent pairs far from each other ( in a proportional way to the distance
    between them inferred from the data )
    """

    # PARAMETERS
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0025)

    num_epochs = epochs

    scale_permutation_adjustion_loss = 1
    scale_reconstruction_loss = 1
    scale_adjacent_distance_loss = 0
    scale_non_adjacent_distance_loss = 0.1

    adjacent_sample_size = 100
    non_adjacent_sample_size = 100
    permutation_datapoints_samples = 64

    epoch_average_loss = 0
    reconstruction_average_loss = 0
    adjacent_average_loss = 0
    non_adjacent_average_loss = 0
    permuted_data_average_loss = 0

    epoch_print_rate = 250
    DISTANCE_CONSTANT = 0.1

    train_data_names = storage_raw.get_sensor_data_names()
    storage_raw.build_permuted_data_random_rotations()
    train_data = array_to_tensor(np.array(storage_raw.get_pure_permuted_raw_env_data()))

    best_loss = 10000000
    stagnation_streak = 0

    for epoch in range(num_epochs):
        if (epoch % 10 == 0):
            storage_raw.build_permuted_data_random_rotations()
            train_data = array_to_tensor(np.array(storage_raw.get_pure_permuted_raw_env_data()))

        reconstruction_loss = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()

        # PERMUTATION ADJUSTION LOSS
        permutation_adjustion_loss = permutation_adjustion_handling(autoencoder, permutation_datapoints_samples,
                                                                    scale_permutation_adjustion_loss)
        permutation_adjustion_loss.backward()

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, criterion, scale_reconstruction_loss)
        reconstruction_loss.backward()

        # ADJACENT DISTANCE LOSS
        adjacent_distance_loss = torch.tensor(0.0)
        # adjacent_distance_loss, average_distance_adjacent = adjacent_distance_handling(autoencoder,
        #                                                                                adjacent_sample_size,
        #                                                                                scale_adjacent_distance_loss)
        # adjacent_distance_loss.backward()

        # NON-ADJACENT DISTANCE LOSS
        non_adjacent_distance_loss = torch.tensor(0.0)
        non_adjacent_distance_loss = non_adjacent_distance_handling(autoencoder, non_adjacent_sample_size,
                                                                    scale_non_adjacent_distance_loss,
                                                                    distance_factor=DISTANCE_CONSTANT)
        non_adjacent_distance_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + adjacent_distance_loss.item() + non_adjacent_distance_loss.item() + permutation_adjustion_loss.item()

        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        adjacent_average_loss += adjacent_distance_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()
        permuted_data_average_loss += permutation_adjustion_loss.item()

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            adjacent_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate
            permuted_data_average_loss /= epoch_print_rate

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
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss} | PERMUTATION LOSS:{permuted_data_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            adjacent_average_loss = 0
            non_adjacent_average_loss = 0

    return autoencoder


def run_ai():
    global storage_raw

    autoencoder = AutoencoderPostPermutor()

    train_autoencoder_with_distance_constraint(autoencoder, epochs=5000)
    return autoencoder


def eval_perm_pre_and_post(autoencoder: BaseAutoencoderModel):
    global storage_raw

    datapoint: List[str] = storage.sample_n_random_datapoints(64)
    datapoints_data = [storage.get_datapoint_data_tensor_by_name(name) for name in datapoint]
    accumulated_loss = torch.tensor(0.0)

    distance_pre = 0
    distance_post = 0
    for datapoint_data in datapoints_data:
        preenc = datapoint_data
        postenc = autoencoder.encoder_training(datapoint_data)
        distance_pre += torch.cdist(preenc, preenc, p=2).mean()
        distance_post += torch.cdist(postenc, postenc, p=2).mean()

    print(f"Pre-enc distance: {distance_pre / len(datapoints_data)}")
    print(f"Post-enc distance: {distance_post / len(datapoints_data)}")


def run_tests(autoencoder):
    global storage_raw

    evaluate_reconstruction_error_super(autoencoder, storage)
    avg_distance_adj = evaluate_distances_between_pairs_super(autoencoder, storage)
    evaluate_adjacency_properties_super(autoencoder, storage, avg_distance_adj)
    eval_perm_pre_and_post(autoencoder)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("autoenc_dynamic10k.pth")
    autoencoder = load_manually_saved_ai("autoencodPerm10k.pth")
    global storage_raw
    storage.build_permuted_data_raw_with_thetas()

    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = run_ai()
    save_ai_manually("autoencodPerm_linearizer", autoencoder)
    run_tests(autoencoder)


def run_permuted_autoencoder2() -> None:
    global storage_raw
    global permutor

    permutor = load_manually_saved_ai("permutor_deshift_working.pth")
    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()
    storage.tanh_all_data()
    storage.set_permutor(permutor)
    storage.build_permuted_data_raw_with_thetas()

    run_new_ai()
    # run_loaded_ai()


storage_raw: StorageSuperset2 = StorageSuperset2()
permutor = None
