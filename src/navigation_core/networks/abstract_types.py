from dataclasses import dataclass


@dataclass
class NetworkTrainingData:
    def __post_init__(self):
        pass


@dataclass
class NetworkTrainingParams:
    epochs_count = 1000
    epoch_print_rate = 100
    stop_at_threshold = False
