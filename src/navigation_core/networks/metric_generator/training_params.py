from dataclasses import dataclass


@dataclass
class MetricTrainingParams:
    epochs = 1000
    epoch_print_rate = 100
    stop_at_threshold = False
    walking_samples = 1000
