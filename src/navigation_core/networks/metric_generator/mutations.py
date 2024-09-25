from src.navigation_core.networks.common import Mutation
from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData, WalkingMetadata
import random
from typing import List, Callable, TYPE_CHECKING
from src import runtime_storages as runtime_storage

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


def build_mutation_function(storage_struct: 'StorageStruct',
                            mutation: Callable[['StorageStruct', MetricTrainingData], None]) -> Callable[
    [MetricTrainingData], None]:
    def mutation_function(training_data: MetricTrainingData) -> None:
        mutation(storage_struct, training_data)

    return mutation_function


def mutate_walking_data_rotations(storage_struct: 'StorageStruct', training_data: 'MetricTrainingData') -> None:
    walking_start_metadata: List[WalkingMetadata] = training_data.walking_batch_start_metadata
    walking_end_metadata: List[WalkingMetadata] = training_data.walking_batch_end_metadata

    walking_start = training_data.walking_batch_start
    walking_end = training_data.walking_batch_end

    for i in range(len(walking_start)):
        start_count_rotations = runtime_storage.node_get_datapoints_count(storage_struct,
                                                                          walking_start_metadata[i].name)
        end_count_rotations = runtime_storage.node_get_datapoints_count(storage_struct, walking_end_metadata[i].name)

        start_new_index = random.randint(0, start_count_rotations - 1)
        if start_new_index == walking_start_metadata[i].datapoint_index:
            start_new_index = (start_new_index + 1) % start_count_rotations
        end_new_index = random.randint(0, end_count_rotations - 1)
        if end_new_index == walking_end_metadata[i].datapoint_index:
            end_new_index = (end_new_index + 1) % end_count_rotations
        start_name = walking_start_metadata[i].name
        end_name = walking_end_metadata[i].name

        start_new_data = runtime_storage.node_get_datapoint_tensor_at_index(storage_struct, start_name, start_new_index)
        end_new_data = runtime_storage.node_get_datapoint_tensor_at_index(storage_struct, end_name, end_new_index)

        walking_start[i] = start_new_data
        walking_end[i] = end_new_data
        walking_start_metadata[i].datapoint_index = start_new_index
        walking_end_metadata[i].datapoint_index = end_new_index
