from typing import TYPE_CHECKING, List
import torch

if TYPE_CHECKING:
    from src.runtime_storages import StorageStruct
    from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData

from src import runtime_storages as storage


def build_rotations_array(storage_struct: 'StorageStruct', training_data: MetricTrainingData) -> List[torch.Tensor]:
    rotations_array = []
    node_names = storage.nodes_get_all_names(storage_struct)
    for node_name in node_names:
        node_data = storage.node_get_datapoints_tensor(storage_struct, node_name)
        rotations_array.append(node_data)

    return rotations_array
