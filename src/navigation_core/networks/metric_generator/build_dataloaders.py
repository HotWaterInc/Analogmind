from typing import TYPE_CHECKING
from torch.utils.data import DataLoader, Dataset
import itertools
from src.navigation_core.networks.metric_generator.types import WalkData, RotationData
from src.utils.utils import get_debug, get_testing

if TYPE_CHECKING:
    from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData
    from src.navigation_core.networks.metric_generator.training_params import MetricTrainingParams


def build_dataloader_walking(training_data_struct: 'MetricTrainingData',
                             training_params: 'MetricTrainingParams') -> None:
    walking_batch_start = training_data_struct.walking_batch_start
    walking_batch_end = training_data_struct.walking_batch_end
    walking_batch_distance = training_data_struct.walking_batch_distance

    dataset = WalkData(walking_batch_start, walking_batch_end, walking_batch_distance)
    dataloader = DataLoader(dataset, batch_size=training_params.walking_samples, shuffle=(not get_testing()) & True)
    dataloader = itertools.cycle(dataloader)
    training_data_struct.walking_dataloader = iter(dataloader)


def build_dataloader_rotations(training_data_struct: 'MetricTrainingData',
                               training_params: 'MetricTrainingParams') -> None:
    rotations_array = training_data_struct.rotations_array
    dataset = RotationData(rotations_array)
    dataloader = DataLoader(dataset, batch_size=training_params.rotations_samples, shuffle=(not get_testing()) & True)
    infinite_dataloader = itertools.cycle(dataloader)

    training_data_struct.rotations_dataloader = iter(infinite_dataloader)
