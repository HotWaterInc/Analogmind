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
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display
import torch
import torch.nn as nn
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm, _make_layer
from src.utils import get_device


class SeenNetwork(BaseAutoencoderModel):
    def __init__(self, dropout_rate: float = 0.2, embedding_size: int = 64, input_output_size: int = 512,
                 hidden_size: int = 256, num_blocks: int = 1):
        super(SeenNetwork, self).__init__()
        self.embedding_size = embedding_size

        self.input_layer = nn.Linear(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.manifold_encoder = _make_layer(hidden_size, embedding_size)
        self.manifold_decoder = _make_layer(embedding_size, hidden_size)

        self.decoding_blocks = nn.ModuleList(
            [ResidualBlockSmallBatchNorm(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.output_layer = nn.Linear(hidden_size, input_output_size)
        self.leaky_relu = nn.LeakyReLU()

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


def reconstruction_handling(autoencoder: SeenNetwork, data: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    dec = autoencoder.forward_training(data)
    criterion = nn.MSELoss()

    return criterion(dec, data) * scale_reconstruction_loss


def _train_seen_network(autoencoder: SeenNetwork, epochs: int,
                        storage: StorageSuperset2,
                        pretty_print: bool = False) -> SeenNetwork:
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0005, amsgrad=True)
    num_epochs = epochs

    scale_reconstruction_loss = 1
    epoch_average_loss = 0
    reconstruction_average_loss = 0

    epoch_print_rate = 250

    autoencoder = autoencoder.to(get_device())
    train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data())).to(get_device())

    if pretty_print:
        set_pretty_display(epoch_print_rate, "Epoch batch")
        pretty_display_start(0)

    for epoch in range(num_epochs):
        reconstruction_loss = torch.tensor(0.0)
        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, scale_reconstruction_loss)
        reconstruction_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item()
        reconstruction_average_loss += reconstruction_loss.item()
        epoch_average_loss += epoch_loss

        if pretty_print:
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
            print("--------------------------------------------------")

            epoch_average_loss = 0
            reconstruction_average_loss = 0

            if pretty_print:
                pretty_display_reset()
                pretty_display_start(epoch)

    return autoencoder


def generate_new_ai() -> SeenNetwork:
    return SeenNetwork()


def run_seen_network(seen_network: SeenNetwork, storage: StorageSuperset2) -> SeenNetwork:
    storage.build_permuted_data_random_rotations_rotation0()
    seen_network = _train_seen_network(seen_network, 2500, storage, True)
    return seen_network


def load_storage_data(storage: StorageSuperset2) -> StorageSuperset2:
    dataset_grid = 5
    storage.load_raw_data_from_others(f"data{dataset_grid}x{dataset_grid}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{dataset_grid}x{dataset_grid}_connections.json")
    # selects first rotation
    storage.build_permuted_data_random_rotations_rotation0()
    return storage
