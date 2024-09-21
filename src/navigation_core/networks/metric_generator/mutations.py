from src.navigation_core.networks.common import Mutation
from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData, WalkingMetadata
from src.navigation_core.params import ROTATIONS
from src.runtime_storages import StorageStruct
import random
from typing import List, Callable
from src import runtime_storages as runtime_storage


def build_mutation_function(storage_struct: 'StorageStruct',
                            mutation: Callable[[StorageStruct, MetricTrainingData], None]) -> Callable[
    [MetricTrainingData], None]:
    def mutation_function(training_data: MetricTrainingData) -> None:
        mutation(storage_struct, training_data)

    return mutation_function


def mutate_rotations(storage_struct: 'StorageStruct', training_data: 'MetricTrainingData') -> None:
    walking_start_metadata: List[WalkingMetadata] = training_data.walking_batch_start_metadata
    walking_end_metadata: List[WalkingMetadata] = training_data.walking_batch_end_metadata

    walking_start = training_data.walking_batch_start
    walking_end = training_data.walking_batch_end

    for i in range(len(walking_start)):
        start_new_index = random.randint(0, ROTATIONS - 1)
        end_new_index = random.randint(0, ROTATIONS - 1)
        start_name = walking_start_metadata[i].name
        end_name = walking_end_metadata[i].name

        start_new_data = runtime_storage.node_get_datapoint_tensor_at_index(storage_struct, start_name, start_new_index)
        end_new_data = runtime_storage.node_get_datapoint_tensor_at_index(storage_struct, end_name, end_new_index)

        walking_start[i] = start_new_data
        walking_end[i] = end_new_data
        walking_start_metadata[i].datapoint_index = start_new_index
        walking_end_metadata[i].datapoint_index = end_new_index
