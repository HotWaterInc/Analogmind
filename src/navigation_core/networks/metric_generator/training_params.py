from dataclasses import dataclass

from src.navigation_core.networks.abstract_types import NetworkTrainingParams


@dataclass
class MetricTrainingParams(NetworkTrainingParams):
    walking_samples = 1000
    rotations_samples = 1000
