from dataclasses import dataclass, fields, field
from typing import List, TYPE_CHECKING, Tuple
import torch
from typing_extensions import Iterator

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


@dataclass
class MetricTrainingData:
    rotations_array: List[torch.Tensor] = field(default_factory=list)

    walking_batch_start: List[torch.Tensor] = field(default_factory=list)
    walking_batch_end: List[torch.Tensor] = field(default_factory=list)
    walking_batch_distance: List[float] = field(default_factory=list)

    walking_batch_start_metadata: List[any] = field(default_factory=list)
    walking_batch_end_metadata: List[any] = field(default_factory=list)

    walking_dataloader: Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]] = None
    rotations_dataloader: Iterator[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]] = None

    def __post_init__(self):
        pass
