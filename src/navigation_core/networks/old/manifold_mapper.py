import torch.optim as optim
import numpy as np
from src.navigation_core import BaseAutoencoderModel
from src.navigation_core import MANIFOLD_SIZE, IMAGE_EMBEDDING_SIZE
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_storages.storage_superset2 import StorageSuperset2
from typing import List
from src.utils import array_to_tensor, get_device
from src.utils.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, pretty_display_set
import torch
import torch.nn as nn
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer_no_batchnorm_leaky


class ManifoldMapper(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, embedding_size: int = MANIFOLD_SIZE,
                 input_output_size: int = IMAGE_EMBEDDING_SIZE,
                 hidden_size: int = 1024 * 4, num_blocks: int = 1):
        super(ManifoldMapper, self).__init__()
        self.embedding_size = embedding_size

        self.input_layer = nn.Linear(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.manifold_encoder = _make_layer_no_batchnorm_leaky(hidden_size, embedding_size)

        self.manifold_decoder = _make_layer_no_batchnorm_leaky(embedding_size, hidden_size)

        self.decoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.output_layer = nn.Linear(hidden_size, input_output_size)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder_training(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.encoding_blocks:
            x = block(x)
        x = self.manifold_encoder(x)

        return x

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_training(x)

    def decoder_training(self, x: torch.Tensor) -> torch.Tensor:
        x = self.manifold_decoder(x)
        for block in self.decoding_blocks:
            x = block(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

    def decoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(x)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_training(x)
        decoded = self.decoder_training(encoded)
        return decoded

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)

    def get_embedding_size(self) -> int:
        return self.embedding_size


def reconstruction_handling(manifold_mapper: BaseAutoencoderModel, storage: StorageSuperset2) -> torch.Tensor:
    embedding_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(get_device())

    manifold_data = storage.get_other_data("manifold")
    manifold_data = [array_to_tensor(datapoint["data"][0]) for datapoint in manifold_data]

    # embedding_data = torch.stack(embedding_data).to(get_device())
    manifold_data = torch.stack(manifold_data).to(get_device())
    manifold_representation_mean = torch.mean(manifold_data, dim=0)
    manifold_absolute_mean = torch.mean(torch.abs(manifold_representation_mean), dim=0)

    enc = manifold_mapper.encoder_training(embedding_data)

    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()

    loss = criterion(enc, manifold_data) + criterion2(enc, manifold_data)
    loss /= manifold_absolute_mean
    return loss


def permutation_adjustion_handling(autoencoder: BaseAutoencoderModel, storage: StorageSuperset2,
                                   samples: int) -> torch.Tensor:
    """
    Keeps the permutation of the data points close to each other
    """
    datapoints: List[str] = storage.sample_n_random_datapoints(samples)
    datapoints_data = [storage.node_get_datapoints_tensor(name) for name in datapoints]
    accumulated_loss = torch.tensor(0.0, device=get_device())

    datapoints_data = torch.stack(datapoints_data).to(get_device())

    original_shape = datapoints_data.shape
    to_encode = datapoints_data.view(-1, original_shape[-1])
    encoded = autoencoder.encoder_training(to_encode)
    encoded = encoded.view(original_shape[0], original_shape[1], encoded.shape[-1])
    accumulated_loss = torch.cdist(encoded, encoded, p=2).mean()

    return accumulated_loss


def _train_manifold_mapper(manifold_network: BaseAutoencoderModel, storage: StorageSuperset2,
                           epochs: int, stop_at_threshold: bool = False) -> BaseAutoencoderModel:
    # PARAMETERS
    optimizer = optim.Adam(manifold_network.parameters(), lr=0.0005)

    num_epochs = epochs

    scale_reconstruction_loss = 1

    epoch_average_loss = 0

    reconstruction_average_loss = 0
    non_adjacent_average_loss = 0
    adjacent_average_loss = 0
    permutation_average_loss = 0

    epoch_print_rate = 500
    DISTANCE_SCALING_FACTOR = 1
    EMBEDDING_SCALING_FACTOR = 0.1

    storage.build_permuted_data_random_rotations_rotation0()
    manifold_network = manifold_network.to(get_device())

    pretty_display_set(epoch_print_rate, "Epoch batch")
    pretty_display_start(0)

    if stop_at_threshold:
        num_epochs = int(1e7)

    print("")
    SHUFFLE_RATE = 2
    for epoch in range(num_epochs):
        if epoch % SHUFFLE_RATE == 0:
            storage.build_permuted_data_random_rotations()
            # storage.build_permuted_data_random_rotations_rotation0()
            pass

        reconstruction_loss = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()
        accumulated_loss = torch.tensor(0.0, device=get_device())

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(
            manifold_mapper=manifold_network,
            storage=storage
        )

        accumulated_loss = reconstruction_loss
        accumulated_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item()
        epoch_average_loss += epoch_loss

        reconstruction_average_loss += reconstruction_loss.item()
        pretty_display(epoch % epoch_print_rate)

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print("")
            print(f"EPOCH:{epoch}/{num_epochs}")
            print(
                f"RECONSTRUCTION LOSS:{reconstruction_average_loss}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("-------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0

            pretty_display_reset()
            pretty_display_start(epoch)

    return manifold_network


def train_manifold_mapper(manifold_mapper: BaseAutoencoderModel, storage: StorageSuperset2):
    manifold_mapper = _train_manifold_mapper(
        manifold_network=manifold_mapper,
        storage=storage,
        epochs=10000,
        stop_at_threshold=False
    )

    return manifold_mapper
