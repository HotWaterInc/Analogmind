import numpy as np
import torch
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration_normal.seen_network import SeenNetwork
from src.utils import array_to_tensor, get_device
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from typing import List
from src.modules.time_profiler import start_profiler, profiler_checkpoint, profiler_checkpoint_blank
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display
from src.ai.variants.exploration_normal.utils import get_missing_connections_based_on_distance


def build_missing_connections_with_cheating(storage: StorageSuperset2, new_datapoints, distance_threshold):
    THRESHOLD = distance_threshold
    print("start build connections")

    all_new_connections = []
    for new_datapoint in new_datapoints:
        new_connections = get_missing_connections_based_on_distance(storage, new_datapoint, THRESHOLD)
        all_new_connections.extend(new_connections)

    storage.incorporate_new_data([], all_new_connections)
    print("finished building connections")


def build_missing_connections_with_metric(storage: StorageSuperset2, metric, new_datapoints: List[any],
                                          distance_threshold):
    """
    Evaluate new datapoints and old datapoints with the distance metric
    """

    THRESHOLD = distance_threshold
    should_be_found = []
    should_not_be_found = []

    print("Started evaluating metric")
    # finds out what new datapoints should be found as adjacent
    for new_datapoint in new_datapoints:
        minimum_distance = check_min_distance(storage, new_datapoint)
        if minimum_distance < THRESHOLD:
            should_be_found.append(new_datapoint)
        else:
            should_not_be_found.append(new_datapoint)

    print("calculated min distances")

    # finds out datapoints by metric
    found_datapoints = []
    negative_datapoints = []

    set_pretty_display(len(new_datapoints), "Distance metric evaluation")
    pretty_display_start()
    for idx, new_datapoint in enumerate(new_datapoints):
        if metric(storage, new_datapoint) == 1:
            found_datapoints.append(new_datapoint)
        else:
            negative_datapoints.append(new_datapoint)

        if idx % 10 == 0:
            pretty_display(idx)

    pretty_display_reset()

    print("calculated metric results")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    false_positives_arr = []

    for found_datapoint in found_datapoints:
        if found_datapoint in should_be_found:
            true_positives += 1
        else:
            false_positives += 1
            false_positives_arr.append(found_datapoint)

    for negative_datapoint in negative_datapoints:
        if negative_datapoint in should_not_be_found:
            true_negatives += 1
        else:
            false_negatives += 1

    if len(found_datapoints) == 0:
        print("No found datapoints for this metric")
        return

    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"True negatives: {true_negatives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {true_positives / (true_positives + false_positives)}")
    print(f"Recall: {true_positives / (true_positives + false_negatives)}")
    print(
        f"Accuracy: {(true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)}")

    for false_positive in false_positives_arr:
        distance = check_min_distance(storage, false_positive)
        print("false positive", distance)
