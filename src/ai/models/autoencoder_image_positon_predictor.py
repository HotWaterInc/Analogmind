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
    evaluate_adjacency_properties_super, evaluate_reconstruction_error_super_fist_rotation, \
    evaluate_differences_between_rotations
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display

import torch
import torch.nn as nn


class ResidualBlockSmall(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlockSmall, self).__init__()
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


def _make_layer(in_features, out_features):
    layer = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )
    return layer


class AutoencoderPositionalBlock(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, position_embedding_size: int = 16,
                 hidden_size: int = 128, num_blocks: int = 1, input_size=512, output_size=2, position_range=3,
                 ):
        super(AutoencoderPositionalBlock, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.encoding_blocks = nn.ModuleList([ResidualBlockSmall(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.positional_encoder_last = _make_layer(hidden_size, position_embedding_size)
        self.decoder_initial_layer = _make_layer(position_embedding_size, hidden_size)

        self.decoding_blocks = nn.ModuleList([ResidualBlockSmall(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.leaky_relu = nn.LeakyReLU()
        self.embedding_size = position_embedding_size
        self.tanh = nn.Tanh()
        self.position_range = position_range

    def encoder_training(self, x: torch.Tensor) -> any:
        initial_data = x

        x = self.input_layer(x)
        for block in self.encoding_blocks:
            x = block(x)

        positional_encoding = self.positional_encoder_last(x)
        return positional_encoding

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        position = self.encoder_training(x)
        return position

    def decoder_training(self, positional_encoding) -> torch.Tensor:
        x = self.decoder_initial_layer(positional_encoding)
        for block in self.decoding_blocks:
            x = block(x)

        x = self.output_layer(x)
        x = self.tanh(x) * self.position_range
        return x

    def decoder_inference(self, positional_encoding, rotational_encoding) -> torch.Tensor:
        return self.decoder_training(positional_encoding)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        positional_encoding = self.encoder_training(x)
        return self.decoder_training(positional_encoding)

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)

    def get_embedding_size(self) -> int:
        return self.embedding_size


def position_reconstruction_handling(autoencoder: BaseAutoencoderModel, data: any, positions_data: any,
                                     scale_reconstruction_loss: int = 1) -> torch.Tensor:
    dec = autoencoder.forward_training(data)
    criterion = nn.MSELoss()

    return criterion(dec, positions_data) * scale_reconstruction_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def permutation_adjustion_handling(autoencoder: BaseAutoencoderModel, samples: int,
                                   scale_permutation_adjustion_loss: float) -> torch.Tensor:
    """
    Keeps the permutation of the data points close to each other
    """
    global storage

    datapoint: List[str] = storage.sample_n_random_datapoints(samples)
    datapoints_data = [storage.get_datapoint_data_tensor_by_name(name).to(device) for name in datapoint]
    accumulated_loss = torch.tensor(0.0, device=device)
    autoencoder = autoencoder.to(device)

    for datapoint_data in datapoints_data:
        enc = autoencoder.encoder_training(datapoint_data)
        loss = torch.cdist(enc, enc, p=2).mean()
        accumulated_loss += loss

    return accumulated_loss / samples * scale_permutation_adjustion_loss


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

    batch_datapoint1 = torch.stack(batch_datapoint1).to(device)
    batch_datapoint2 = torch.stack(batch_datapoint2).to(device)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] * distance_per_neuron * embedding_size for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance).to(device)

    criterion = nn.MSELoss()
    non_adjacent_distance_loss = criterion(distance, expected_distance) * scale_non_adjacent_distance_loss
    return non_adjacent_distance_loss


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

    batch_datapoint1 = torch.stack(batch_datapoint1).to(device)
    batch_datapoint2 = torch.stack(batch_datapoint2).to(device)

    encoded_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j = autoencoder.encoder_training(batch_datapoint2)

    # embedding_size = encoded_i.shape[1]

    distance = torch.sum(torch.norm((encoded_i - encoded_j), p=2)).to("cpu")

    average_distance += distance.item() / adjacent_sample_size
    adjacent_distance_loss += distance / adjacent_sample_size * scale_adjacent_distance_loss

    return adjacent_distance_loss, average_distance


def minimize_embeddings_handler(autoencoder: BaseAutoencoderModel, data: any) -> torch.Tensor:
    """
    Minimizes the embeddings
    """
    enc = autoencoder.encoder_training(data)
    return torch.norm(enc, p=2) / enc.shape[0]


def train_autoencoder_position_predictor(autoencoder: BaseAutoencoderModel, epochs: int,
                                         pretty_print: bool = False) -> nn.Module:
    # PARAMETERS
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.005, amsgrad=True)

    num_epochs = epochs

    scale_reconstruction_loss = 1
    scale_permutation_loss = 5
    scale_non_adjacent_distance_loss = 1

    scale_minimum_embedding_loss = 0
    scale_adjacent_loss = 0.4

    epoch_average_loss = 0

    reconstruction_average_loss = 0
    permutation_average_loss = 0
    minimize_average_embeddings_loss = 0

    non_adjacent_distance_average_loss = 0
    adjacent_distance_average_loss = 0

    permutation_samples = 5 * 5
    non_adjacent_samples = 300
    adjacent_samples = 5 * 5

    epoch_print_rate = 250

    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))
    position_data = array_to_tensor(np.array(storage.get_pure_xy_permuted_raw_env_data()))

    SHUFFLE_RATE = 5
    DISTANCE_PER_NEURON = 0.1

    if pretty_print:
        set_pretty_display(epoch_print_rate, "Epoch batch")
        pretty_display_start(0)

    autoencoder = autoencoder.to(device)

    for epoch in range(num_epochs):
        if (epoch % SHUFFLE_RATE == 0):
            # storage.build_permuted_data_random_rotations()
            storage.build_permuted_data_random_rotations_rotation0()

            train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(device=device)
            position_data = array_to_tensor(np.array(storage.get_pure_xy_permuted_raw_env_data())).to(device=device)

        reconstruction_loss = torch.tensor(0.0)
        permutation_loss = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        # reconstruction_loss = position_reconstruction_handling(autoencoder, train_data, position_data,
        #                                                        scale_reconstruction_loss)
        # reconstruction_loss.backward()

        # PERMUTATION LOSS
        permutation_loss = permutation_adjustion_handling(autoencoder, permutation_samples, scale_permutation_loss)
        permutation_loss.backward()
        # NON ADJACENCY LOSS
        non_adjacent_distance_loss = non_adjacent_distance_handling(autoencoder, non_adjacent_samples,
                                                                    scale_non_adjacent_distance_loss,
                                                                    DISTANCE_PER_NEURON)
        non_adjacent_distance_loss.backward()

        # MINIMIZE EMBEDDINGS ( KL LIKE )
        minimize_embeddings = torch.tensor(0.0)
        # minimize_embeddings = minimize_embeddings_handler(autoencoder, train_data)
        minimize_embeddings *= scale_minimum_embedding_loss
        # minimize_embeddings.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + permutation_loss.item() + minimize_embeddings.item() + non_adjacent_distance_loss.item()

        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        permutation_average_loss += permutation_loss.item()
        minimize_average_embeddings_loss += minimize_embeddings.item()
        non_adjacent_distance_average_loss += non_adjacent_distance_loss.item()

        if pretty_print:
            pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:

            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            permutation_average_loss /= epoch_print_rate
            minimize_average_embeddings_loss /= epoch_print_rate
            non_adjacent_distance_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs} ")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | PERMUTATION LOSS:{permutation_average_loss} | MINIMIZE EMBEDDINGS LOSS:{minimize_average_embeddings_loss} | NON ADJACENT DISTANCE LOSS:{non_adjacent_distance_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0
            permutation_average_loss = 0
            non_adjacent_distance_average_loss = 0
            adjacent_distance_average_loss = 0
            minimize_average_embeddings_loss = 0

            if pretty_print:
                pretty_display_reset()
                pretty_display_start(epoch)

    return autoencoder


def run_ai():
    global storage
    autoencoder = AutoencoderPositionalBlock()
    autoencoder = train_autoencoder_position_predictor(autoencoder, epochs=5001, pretty_print=True)
    return autoencoder


def run_tests(autoencoder):
    global storage

    evaluate_differences_between_rotations(autoencoder, storage)
    # evaluate_reconstruction_error_super_fist_rotation(autoencoder, storage)
    # evaluate_reconstruction_error_super(autoencoder, storage, rotations0=False)
    avg_distance_adj = evaluate_distances_between_pairs_super(autoencoder, storage, rotations0=True)
    evaluate_adjacency_properties_super(autoencoder, storage, avg_distance_adj, rotation0=True)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("autoenc_dynamic10k.pth")
    autoencoder = load_manually_saved_ai("autoencod_position_predictor.pth")
    global storage
    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = run_ai()
    save_ai_manually("autoencod_position_predictor", autoencoder)
    run_tests(autoencoder)


def run_autoencoder_position_predictor() -> None:
    global storage

    grid_data = 5

    storage.load_raw_data_from_others(f"data{grid_data}x{grid_data}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_data}x{grid_data}_connections.json")
    # selects first rotation
    storage.build_permuted_data_random_rotations_rotation0()

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = StorageSuperset2()
