from torch import nn
from abc import ABC, abstractmethod
import torch


class MetricNetworkAbstract(nn.Module, ABC):
    def __init__(self):
        super(MetricNetworkAbstract, self).__init__()

    @abstractmethod
    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decoder_inference(self, *args) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encoder_training(self, *args) -> any:
        pass

    @abstractmethod
    def decoder_training(self, *args) -> any:
        pass

    @abstractmethod
    def forward_training(self, *args) -> any:
        pass

    @abstractmethod
    def get_embedding_size(self) -> int:
        pass
