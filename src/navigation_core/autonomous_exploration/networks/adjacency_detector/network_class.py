from src.navigation_core.networks.blocks import BlockResidualSmall, LayerLeaky
from src.navigation_core.networks.metric_generator.metric_network_abstract import MetricNetworkAbstract
from src.navigation_core.to_refactor.params import MANIFOLD_SIZE

import torch
from torch import nn
from torch.nn import functional as F


class AdjacencyDetector(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=2, dropout_rate=0.6, num_blocks=1):
        super(AdjacencyDetector, self).__init__()
        self.input_layer = LayerLeaky(input_size * 2, hidden_size)
        self.blocks = nn.ModuleList([BlockResidualSmall(hidden_size, dropout_rate) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def _forward_pass(self, x, y):
        inpt = torch.cat((x, y), dim=1)

        out = self.input_layer(inpt)
        for block in self.blocks:
            out = block(out)

        output = self.output_layer(out)
        output = F.softmax(output, dim=1)

        return output

    def forward_training(self, x, y):
        output = self._forward_pass(x, y)
        return output

    def forward(self, x, y):
        output = self._forward_pass(x, y)
        return output
