from torch import nn
from abc import ABC, abstractmethod
import torch


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

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
