import torch
import time
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer_no_batchnorm_leaky, _make_layer, \
    _make_layer_relu, _make_layer_no_batchnorm_relu, _make_layer_linear, ResidualBlockSmallBatchNormWithAttention
from src.ai.variants.exploration.utils import ROTATIONS, ROTATIONS_PER_FULL, ROTATIONS, OFFSETS_PER_DATAPOINT, \
    THETAS_SIZE
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_storages.storage_superset2 import StorageSuperset2, \
    angle_percent_to_thetas_normalized_cached
from src.ai.runtime_storages import Storage
from typing import List, Dict, Union
from src.modules.time_profiler import profiler_checkpoint_blank, start_profiler, profiler_checkpoint
from src.utils import array_to_tensor, get_device
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super, evaluate_reconstruction_error_super_fist_rotation
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, pretty_display_set
import torch
import torch.nn as nn


class AbstractionBlockSimplified(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, position_embedding_size: int = 96,
                 thetas_embedding_size: int = THETAS_SIZE,
                 hidden_size: int = 512, num_blocks: int = 1, input_output_size=512 * ROTATIONS_PER_FULL):
        super(AbstractionBlockSimplified, self).__init__()

        self.input_layer = _make_layer(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNormWithAttention(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.positional_encoder = _make_layer(hidden_size, position_embedding_size)

        self.decoder_initial_layer = _make_layer(position_embedding_size + thetas_embedding_size,
                                                 hidden_size)
        self.decoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNormWithAttention(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.output_layer = _make_layer_no_batchnorm_leaky(hidden_size, input_output_size)
        self.embedding_size = position_embedding_size
        self.sigmoid = nn.Sigmoid()

    def encoder_training(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input_layer(x)
        for block in self.encoding_blocks:
            x = block(x)

        position = self.positional_encoder(x)

        return position

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        position = self.encoder_training(x)
        return position

    def decoder_training(self, positional_encoding, rotational_encoding) -> torch.Tensor:
        x = torch.cat([positional_encoding, rotational_encoding], dim=-1)
        x = self.decoder_initial_layer(x)

        for block in self.decoding_blocks:
            x = block(x)

        x = self.output_layer(x)
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


_cache_reconstruction_loss = {}
_cache_thetas = {}


def reconstruction_handling_simplified(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2) -> any:
    global _cache_reconstruction_loss, _cache_thetas
    sampled_count = 100
    sampled_points = storage.sample_n_random_datapoints(sampled_count)

    loss_reconstruction = torch.tensor(0.0, device=get_device())
    loss_cdist = torch.tensor(0.0, device=get_device())

    data_arr = []
    thetas_arr = []
    for point in sampled_points:
        point_data_arr = []
        thetas_data_arr = []
        if point in _cache_reconstruction_loss:
            point_data_arr = _cache_reconstruction_loss[point]
            thetas_data_arr = _cache_thetas[point]
        else:
            # creates list of inputs for that point
            for i in range(OFFSETS_PER_DATAPOINT):
                sampled_full_rotation = array_to_tensor(
                    storage.get_point_rotations_with_full_info_set_offset_concatenated(point, ROTATIONS_PER_FULL, i))
                point_data_arr.append(sampled_full_rotation)
                percent = i / OFFSETS_PER_DATAPOINT
                thetas_data_arr.append(angle_percent_to_thetas_normalized_cached(percent, THETAS_SIZE))

            _cache_reconstruction_loss[point] = point_data_arr
            _cache_thetas[point] = thetas_data_arr

        data_arr.append(point_data_arr)
        thetas_arr.append(thetas_data_arr)

    data_arr_tensor = torch.stack([torch.stack(pda) for pda in data_arr]).to(get_device())
    thetas_arr_tensor = torch.stack([torch.stack(tda) for tda in thetas_arr]).to(get_device())

    # reshape
    original_shape_data = data_arr_tensor.shape
    x_reshaped = data_arr_tensor.view(-1, original_shape_data[2])
    # forward
    positional_encoding = autoencoder.encoder_training(x_reshaped)
    # reshape back
    positional_encoding = positional_encoding.view(original_shape_data[0], original_shape_data[1], -1)
    loss_cdist = torch.cdist(positional_encoding, positional_encoding, p=2).mean()
    # average over each dimension of the embedding
    positional_encoding = positional_encoding.mean(dim=1)
    # duplicate to batch size
    positional_encoding = positional_encoding.to("cpu").detach().numpy()
    positional_encoding = np.repeat(positional_encoding[:, np.newaxis, :], OFFSETS_PER_DATAPOINT, axis=1)
    positional_encoding = array_to_tensor(positional_encoding).to(get_device())
    new_shape_data = positional_encoding.shape

    # reshape again
    positional_encoding = positional_encoding.view(-1, new_shape_data[2])
    # build thetas encoding
    thetas_arr_tensor = thetas_arr_tensor.view(-1, thetas_arr_tensor.shape[2])

    dec = autoencoder.decoder_training(positional_encoding, thetas_arr_tensor)
    criterion = nn.MSELoss()
    loss_reconstruction = criterion(dec, x_reshaped)

    return {
        "loss_reconstruction": loss_reconstruction,
        "loss_cdist": loss_cdist,
    }


def linearity_distance_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                                non_adjacent_sample_size: int,
                                scale_non_adjacent_distance_loss: float, distance_per_neuron: float) -> torch.Tensor:
    """
    Makes first degree connections be linearly distant from each other
    """
    # sampled_pairs = storage.sample_datapoints_adjacencies(non_adjacent_sample_size)
    non_adjacent_sample_size = min(non_adjacent_sample_size, len(storage.get_all_connections_data()))
    sampled_pairs = storage.connections_sample(non_adjacent_sample_size)

    batch_datapoint1 = []
    batch_datapoint2 = []

    for pair in sampled_pairs:
        datapoint1 = storage.get_datapoint_data_tensor_by_name_permuted(pair["start"])
        datapoint2 = storage.get_datapoint_data_tensor_by_name_permuted(pair["end"])

        batch_datapoint1.append(datapoint1)
        batch_datapoint2.append(datapoint2)

    batch_datapoint1 = torch.stack(batch_datapoint1).to(get_device())
    batch_datapoint2 = torch.stack(batch_datapoint2).to(get_device())

    encoded_i, thetas_i = autoencoder.encoder_training(batch_datapoint1)
    encoded_j, thetas_j = autoencoder.encoder_training(batch_datapoint2)

    embedding_size = encoded_i.shape[1]

    distance = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    expected_distance = [pair["distance"] for pair in sampled_pairs]
    expected_distance = torch.tensor(expected_distance, dtype=torch.float32).to(get_device())

    criterion = nn.MSELoss()

    non_adjacent_distance_loss = criterion(distance, expected_distance) * scale_non_adjacent_distance_loss
    return non_adjacent_distance_loss


def _train_autoencoder_abstraction_block(abstraction_block: AbstractionBlockSimplified, storage: StorageSuperset2,
                                         epochs: int,
                                         pretty_print: bool = False) -> AbstractionBlockSimplified:
    # PARAMETERS
    optimizer = optim.Adam(abstraction_block.parameters(), lr=0.00075, amsgrad=True)
    abstraction_block = abstraction_block.to(device=get_device())

    num_epochs = epochs
    scale_reconstruction_loss = 2
    scale_ratio_loss = 1.5
    scale_linearity_loss = 1

    epoch_average_loss = 0
    linearity_sample_size = 100
    DISTANCE_PER_NEURON = 0.005

    cdist_average_loss = 0
    reconstruction_average_loss = 0

    epoch_print_rate = 100

    if pretty_print:
        pretty_display_set(epoch_print_rate, "Epoch batch")
        pretty_display_start(0)

    rotation_encoding_change_on_position_avg = 0
    rotation_encoding_change_on_rotation_avg = 0

    position_encoding_change_on_position_avg = 0
    position_encoding_change_on_rotation_avg = 0

    SHUFFLE = 5
    for epoch in range(num_epochs):
        # if epoch % SHUFFLE == 0:
        #     storage.build_permuted_data_random_rotations()

        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        losses_json = reconstruction_handling_simplified(
            abstraction_block,
            storage)

        reconstruction_loss = losses_json["loss_reconstruction"]
        cdist_loss = losses_json["loss_cdist"] * 0.1

        accumulated_loss = reconstruction_loss + cdist_loss
        accumulated_loss.backward()

        optimizer.step()

        cdist_loss /= 0.1

        epoch_average_loss += reconstruction_loss.item()
        reconstruction_average_loss += reconstruction_loss.item()
        cdist_average_loss += cdist_loss.item()

        if pretty_print:
            pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:

            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            cdist_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs} ")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | CDIST LOSS:{cdist_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0

            if pretty_print:
                pretty_display_reset()
                pretty_display_start(epoch)

    return abstraction_block


def run_abstraction_block_exploration_simplified(abstraction_network: AbstractionBlockSimplified,
                                                 storage: StorageSuperset2) -> AbstractionBlockSimplified:
    abstraction_network = _train_autoencoder_abstraction_block(abstraction_network, storage, 7500, True)
    return abstraction_network
