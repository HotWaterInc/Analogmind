from dataclasses import dataclass


@dataclass
class NetworkTrainingData:
    def __post_init__(self):
        pass


@dataclass
class NetworkTrainingParams:
    epochs_count: int
    epoch_print_rate: int
    stop_at_threshold: bool
    learning_rate: float
