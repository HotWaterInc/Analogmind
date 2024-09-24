from typing import TYPE_CHECKING, List
import torch
import random

from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData, WalkingMetadata

if TYPE_CHECKING:
    from src.runtime_storages import StorageStruct

from src import runtime_storages as storage


def build_rotations_data(storage_struct: 'StorageStruct', training_data: 'MetricTrainingData') -> None:
    rotations_array = []
    node_names = storage.nodes_get_all_names(storage_struct)
    for node_name in node_names:
        node_data = storage.node_get_datapoints_tensor(storage_struct, node_name)
        rotations_array.append(node_data)

    training_data.rotations_array = rotations_array


def build_walking_data(storage_struct: 'StorageStruct', training_data: 'MetricTrainingData') -> None:
    walking_batch_start_names = []
    walking_batch_end_names = []

    walking_batch_distance = []
    walking_batch_start_metadata: List[WalkingMetadata] = []
    walking_batch_end_metadata: List[WalkingMetadata] = []

    node_names = storage.nodes_get_all_names(storage_struct)
    length = len(node_names)

    for i in range(length):
        for j in range(length):
            walking_batch_start_names.append(node_names[i])
            walking_batch_end_names.append(node_names[j])

    # chooses for each node one of the datapoints collected there
    for i in range(length ** 2):
        start_count_rotations = storage.node_get_datapoints_count(storage_struct, walking_batch_start_names[i])
        end_count_rotations = storage.node_get_datapoints_count(storage_struct, walking_batch_end_names[i])

        random_rotation_index_start = random.randint(0, start_count_rotations - 1)
        random_rotation_index_end = random.randint(0, end_count_rotations - 1)

        walking_batch_start_metadata.append(WalkingMetadata(
            name=walking_batch_start_names[i],
            datapoint_index=random_rotation_index_start
        ))
        walking_batch_end_metadata.append(WalkingMetadata(
            name=walking_batch_end_names[i],
            datapoint_index=random_rotation_index_end
        ))

    walking_batch_start = []
    walking_batch_end = []
    for i in range(length ** 2):
        start_name = walking_batch_start_names[i]
        start_index = walking_batch_start_metadata[i].datapoint_index
        end_name = walking_batch_end_names[i]
        end_index = walking_batch_end_metadata[i].datapoint_index

        start_tensor = storage.node_get_datapoint_tensor_at_index(storage_struct, start_name, start_index)
        end_tensor = storage.node_get_datapoint_tensor_at_index(storage_struct, end_name, end_index)
        distance = storage.get_walk_distance(storage_struct, start_name, end_name)

        walking_batch_start.append(start_tensor)
        walking_batch_end.append(end_tensor)
        walking_batch_distance.append(distance)

    training_data.walking_batch_start = walking_batch_start
    training_data.walking_batch_end = walking_batch_end
    training_data.walking_batch_distance = walking_batch_distance
    training_data.walking_batch_start_metadata = walking_batch_start_metadata
    training_data.walking_batch_end_metadata = walking_batch_end_metadata
