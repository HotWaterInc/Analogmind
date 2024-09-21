from dataclasses import dataclass

from src.navigation_core.networks.abstract_types import NetworkTrainingParams


@dataclass
class MetricTrainingParams(NetworkTrainingParams):
    walking_samples: int
    rotations_samples: int


def create_metric_training_params():
    return MetricTrainingParams(epochs_count=100, epoch_print_rate=10, stop_at_threshold=False, walking_samples=100,
                                rotations_samples=100, learning_rate=0.0001)
