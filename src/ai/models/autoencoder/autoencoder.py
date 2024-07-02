import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, load_latest_ai, load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_model import BaseModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties


class Autoencoder(BaseModel):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.Sigmoid()
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
        return decoded

    def decoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(x)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_training(x)
        decoded = self.decoder_training(encoded)
        return decoded

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)


def reconstruction_handling(autoencoder: Autoencoder, data: any, criterion: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    enc = autoencoder.encoder_training(data)
    dec = autoencoder.decoder_training(enc)
    return criterion(dec, data) * scale_reconstruction_loss


def adjacent_distance_handling(autoencoder: Autoencoder, adjacent_sample_size: int,
                               scale_adjacent_distance_loss: float) -> tuple[torch.Tensor, float]:
    """
    Keeps adjacent pairs close to each other
    """
    adjacent_distance_loss = torch.tensor(0.0)
    average_distance = 0
    sampled_pairs = storage.sample_adjacent_datapoints(adjacent_sample_size)
    for pair in sampled_pairs:
        # keep adjacent close to each other
        data_point1 = storage.get_datapoint_data_tensor_by_name(pair["start"])
        data_point2 = storage.get_datapoint_data_tensor_by_name(pair["end"])

        encoded_i = autoencoder.encoder_training(data_point1.unsqueeze(0))
        encoded_j = autoencoder.encoder_training(data_point2.unsqueeze(0))

        distance = torch.norm((encoded_i - encoded_j), p=2)

        average_distance += distance.item()
        adjacent_distance_loss += distance * scale_adjacent_distance_loss

    adjacent_distance_loss /= adjacent_sample_size
    average_distance /= adjacent_sample_size

    return adjacent_distance_loss, average_distance


def non_adjacent_distance_handling(autoencoder: Autoencoder, non_adjacent_sample_size: int,
                                   scale_non_adjacent_distance_loss: float, distance_factor: float) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other
    """
    sampled_pairs = storage.sample_non_adjacent_datapoints(non_adjacent_sample_size)
    non_adjacent_distance_loss = torch.tensor(0.0)

    for pair in sampled_pairs:
        datapoint1 = storage.get_datapoint_data_tensor_by_name(pair["start"])
        datapoint2 = storage.get_datapoint_data_tensor_by_name(pair["end"])

        encoded_i = autoencoder.encoder_training(datapoint1.unsqueeze(0))
        encoded_j = autoencoder.encoder_training(datapoint2.unsqueeze(0))

        distance = torch.norm((encoded_i - encoded_j), p=2)
        expected_distance = pair["distance"] * distance_factor

        non_adjacent_distance_loss += (((
                                                distance - expected_distance) / distance_factor) ** 2) * scale_non_adjacent_distance_loss

    non_adjacent_distance_loss /= non_adjacent_sample_size
    return non_adjacent_distance_loss


def train_autoencoder_with_distance_constraint(autoencoder: Autoencoder, epochs: int):
    """
    Trains the autoencoder with 2 additional losses apart from the reconstruction loss:
    - adjacent distance loss: keeps adjacent pairs close to each other
    - non-adjacent distance loss: keeps non-adjacent pairs far from each other ( in a proportional way to the distance
    between them inferred from the data )
    """

    # PARAMETERS
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.02)

    num_epochs = epochs
    scale_reconstruction_loss = 5
    scale_adjacent_distance_loss = 0.3
    scale_non_adjacent_distance_loss = 0.3

    adjacent_sample_size = 52
    non_adjacent_sample_size = 224

    epoch_average_loss = 0
    reconstruction_average_loss = 0
    adjacent_average_loss = 0
    non_adjacent_average_loss = 0

    epoch_print_rate = 1000
    DISTANCE_CONSTANT = 0.5

    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data()))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        # adjacent_distance_loss = torch.tensor(0.0)
        # non_adjacent_distance_loss = torch.tensor(0.0)

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, criterion, scale_reconstruction_loss)
        reconstruction_loss.backward()

        # ADJACENT DISTANCE LOSS
        adjacent_distance_loss, average_distance_adjacent = adjacent_distance_handling(autoencoder,
                                                                                       adjacent_sample_size,
                                                                                       scale_adjacent_distance_loss)
        adjacent_distance_loss.backward()

        # NON-ADJACENT DISTANCE LOSS
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

            # Print average loss for this epoch
            print(f"EPOCH:{epoch}/{num_epochs}")
            print(f"average distance between adjacent: {average_distance_adjacent}")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

    return autoencoder


def run_ai():
    autoencoder = Autoencoder()
    train_autoencoder_with_distance_constraint(autoencoder, epochs=10)
    return autoencoder


def run_tests(autoencoder):
    global storage

    evaluate_reconstruction_error(autoencoder, storage)
    avg_distance_adj = evaluate_distances_between_pairs(autoencoder, storage)
    evaluate_adjacency_properties(autoencoder, storage, avg_distance_adj)


def run_loaded_ai():
    # autoencoder = load_latest_ai(AIType.Autoencoder)
    # autoencoder = load_manually_saved_ai("autoenc_dynamic10k.pth")
    autoencoder = load_manually_saved_ai("autoencod_static_30k.pth")

    run_tests(autoencoder)
    # run_lee(autoencoder, all_sensor_data, sensor_data)


def run_new_ai() -> None:
    autoencoder = run_ai()
    save_ai("autoencod", AIType.Autoencoder, autoencoder)
    run_tests(autoencoder)


def run_autoencoder() -> None:
    global storage
    storage = Storage()
    storage.load_raw_data(CollectedDataType.Data8x8)
    storage.normalize_all_data()

    run_new_ai()
    # run_loaded_ai()


storage: Storage = None
model = Autoencoder()

if __name__ == '__main__':
    run_autoencoder()
