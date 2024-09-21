from torch import nn
from abc import ABC, abstractmethod
import torch


class NetworkAbstract(nn.Module, ABC):
    def __init__(self):
        super(NetworkAbstract, self).__init__()
