from src.navigation_core.networks.blocks import BlockResidualSmall, LayerLeaky
from src.navigation_core.networks.metric_generator.metric_network_abstract import MetricNetworkAbstract
from src.navigation_core.params import MANIFOLD_SIZE

import torch
from torch import nn


class ManifoldNetwork(MetricNetworkAbstract):
    def __init__(self, dropout_rate: float = 0.2, embedding_size: int = MANIFOLD_SIZE, input_output_size: int = 512,
                 hidden_size: int = 512, num_blocks: int = 2):
        super(ManifoldNetwork, self).__init__()
        self.embedding_size = embedding_size

        self.input_layer = nn.Linear(input_output_size, hidden_size)
        self.encoding_blocks = nn.ModuleList(
            [BlockResidualSmall(hidden_size, dropout_rate) for _ in range(num_blocks)])

        self.manifold_encoder = LayerLeaky(hidden_size, embedding_size)
        self.manifold_decoder = LayerLeaky(embedding_size, hidden_size)

        self.decoding_blocks = nn.ModuleList(
            [BlockResidualSmall(hidden_size, dropout_rate) for _ in range(num_blocks)])

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
