from typing import TYPE_CHECKING
from torch.utils.data import DataLoader, Dataset

from src.navigation_core.networks.metric_generator.types import WalkData

if TYPE_CHECKING:
    from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData
    from src.navigation_core.networks.metric_generator.training_params import MetricTrainingParams


def build_dataloader_walking(training_data_struct: 'MetricTrainingData',
                             training_params: 'MetricTrainingParams') -> None:
    walking_batch_start = training_data_struct.walking_batch_start
    walking_batch_end = training_data_struct.walking_batch_end
    walking_batch_distance = training_data_struct.walking_batch_distance

    dataset = WalkData(walking_batch_start, walking_batch_end, walking_batch_distance)
    dataloader = DataLoader(dataset, batch_size=training_params.walking_samples, shuffle=True)
    training_data_struct.walking_dataloader = iter(dataloader)
