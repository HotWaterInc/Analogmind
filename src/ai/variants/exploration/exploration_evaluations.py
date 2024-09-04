import numpy as np
import torch
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.networks.seen_network import SeenNetwork
from src.utils import array_to_tensor, get_device
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from typing import List
from src.modules.time_profiler import start_profiler, profiler_checkpoint, profiler_checkpoint_blank
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, pretty_display_set
import time
import math
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.pretty_display import pretty_display_start, pretty_display_set, pretty_display
from src.modules.save_load_handlers.data_handle import write_other_data_to_file, serialize_object_other, \
    deserialize_object_other
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import *
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
import torch
from src.utils import get_device


def evaluation_distance_ground_truth_metric(storage: StorageSuperset2, new_datapoint: Dict[str, any],
                                            lower_bound_distance_threshold,
                                            upper_bound_distance_threshold):
    """
    Find the closest datapoint in the storage
    """
    current_name = new_datapoint["name"]
    datapoints_names = storage.get_all_datapoints()
    adjacent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(current_name, 1)
    adjacent_names.append(current_name)

    current_data_arr = []
    other_datapoints_data_arr = []
    selected_names = []

    for name in datapoints_names:
        if name in adjacent_names or name == current_name:
            continue

        selected_names.append(name)

    found_connections = []
    for name in selected_names:
        distance = storage.get_datapoints_real_distance(current_name, name)
        if distance < upper_bound_distance_threshold and distance > lower_bound_distance_threshold:
            found_connections.append(name)

    return found_connections


def print_datapoints_embeddings_distance(storage: StorageSuperset2, datapoint1, datapoint2):
    """
    Print the distance between two datapoints
    """
    distance = storage.get_datapoints_real_distance(datapoint1, datapoint2)
    embedding1 = storage.get_datapoint_data_selected_rotation_tensor_by_name(datapoint1, 0)
    embedding2 = storage.get_datapoint_data_selected_rotation_tensor_by_name(datapoint2, 0)
    distance_embeddings = torch.norm(embedding1 - embedding2).item()
    print("Distance embeddings ", distance_embeddings)

    # print("Distance between", datapoint1, "and", datapoint2, "is", distance)


def evaluate_distance_metric_on_already_found_connections(storage: StorageSuperset2, new_datapoints: List[any],
                                                          found_connections: List[any]):
    """
    Evaluate new datapoints and old datapoints with the distance metric
    """
    true_positive = 0
    true_positives_distant = 0
    really_bad_false_positive = 0

    # new_datapoints = new_datapoints[:300]

    pretty_display_set(len(new_datapoints))
    pretty_display_start()

    all_found_datapoints = []
    true_found_datapoints = []

    false_positive_distances = []
    avg_distant_true_positive = 0

    hasmap_connections = {}

    for conn in found_connections:
        start_name = conn["start"]
        end_name = conn["end"]
        if start_name not in hasmap_connections:
            hasmap_connections[start_name] = []
        if end_name not in hasmap_connections:
            hasmap_connections[end_name] = []

        if end_name not in hasmap_connections[start_name]:
            hasmap_connections[start_name].append(end_name)
        if start_name not in hasmap_connections[end_name]:
            hasmap_connections[end_name].append(start_name)

    def metric(storage: StorageSuperset2, new_datapoint: Dict[str, any]):
        found_datapoints = []
        current_name = new_datapoint["name"]
        if current_name in hasmap_connections:
            found_datapoints = hasmap_connections[current_name]
        return found_datapoints

    for idx, new_datapoint in enumerate(new_datapoints):
        new_datapoint = storage.get_datapoint_by_name(new_datapoint)
        pretty_display(idx)

        found_datapoints = metric(storage, new_datapoint)
        all_found_datapoints.extend(found_datapoints)

        true_datapoints_lower_end = evaluation_distance_ground_truth_metric(storage, new_datapoint, 0, 0.5)
        true_found_datapoints.extend(true_datapoints_lower_end)

        for founddp in found_datapoints:
            distance = storage.get_datapoints_real_distance(new_datapoint["name"], founddp)

            if distance < 0.5:
                true_positive += 1
            elif distance < 1:
                true_positives_distant += 1
                avg_distant_true_positive += distance
            else:
                really_bad_false_positive += 1
                real_distance = storage.get_datapoints_real_distance(new_datapoint["name"], founddp)
                false_positive_distances.append(real_distance)

    print("")
    print("True positive", true_positive)
    print("True positive distant", true_positives_distant)
    if len(true_found_datapoints) == 0:
        print("No true found datapoints")
    else:
        print("Percent of true positive of actual positives", true_positive / len(true_found_datapoints))
    print("Really bad false positive", really_bad_false_positive)
    print("")
    # print("False positive distances", false_positive_distances)


def check_connection_already_existing(connections_arr, start, end):
    """
    Check if the connection already exists
    """
    for conn in connections_arr:
        if conn["start"] == start and conn["end"] == end:
            return True
        if conn["start"] == end and conn["end"] == start:
            return True
    return False


def evaluate_distance_metric(storage: StorageSuperset2, metric, new_datapoints: List[str], debug: bool = False
                             ):
    """
    Evaluate new datapoints and old datapoints with the distance metric
    """
    true_positive = 0
    true_positives_distant = 0
    really_bad_false_positive = 0

    pretty_display_set(len(new_datapoints))
    pretty_display_start()

    all_found_datapoints = []
    true_found_datapoints = []

    false_positive_distances = []
    avg_distant_true_positive = 0

    new_connections_pairs = []
    for idx, new_datapoint in enumerate(new_datapoints):
        new_datapoint = storage.get_datapoint_by_name(new_datapoint)
        pretty_display(idx)

        found_datapoints = metric(storage, new_datapoint)
        all_found_datapoints.extend(found_datapoints)

        true_datapoints_lower_end = evaluation_distance_ground_truth_metric(storage, new_datapoint, 0, 0.5)
        true_found_datapoints.extend(true_datapoints_lower_end)

        for founddp in found_datapoints:
            if check_connection_already_existing(new_connections_pairs, new_datapoint["name"], founddp):
                continue
            if check_connection_already_existing(storage.get_all_connections_data(), new_datapoint["name"], founddp):
                continue

            distance = storage.get_datapoints_real_distance(new_datapoint["name"], founddp)
            new_connections_pairs.append({
                "start": new_datapoint["name"],
                "end": founddp,
                "distance": None,
                "direction": None
            })

            if distance < 0.5:
                true_positive += 1
            elif distance < 1:
                true_positives_distant += 1
                avg_distant_true_positive += distance
            else:
                really_bad_false_positive += 1
                real_distance = storage.get_datapoints_real_distance(new_datapoint["name"], founddp)
                false_positive_distances.append(real_distance)

    if debug:
        print("")
        print("True positive", true_positive)
        print("True positive distant", true_positives_distant)
        if len(true_found_datapoints) == 0:
            print("No true found datapoints")
        else:
            print("Percent of true positive of actual positives", true_positive / len(true_found_datapoints))
        print("Really bad false positive", really_bad_false_positive)
        print("")

    return new_connections_pairs


def _get_connection_distances_raw_data(storage: StorageSuperset2) -> any:
    connections = storage.get_all_connections_only_datapoints()
    SAMPLES = min(500, len(connections))

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(start_name, 0)
        end_data = storage.get_datapoint_data_tensor_by_name_and_index(end_name, 0)

        start_data_arr.append(start_data)
        end_data_arr.append(end_data)

    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())

    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distance_embeddings = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)

    length = len(distance_embeddings)

    for i in range(length):
        start_name = sampled_connections[i]["start"]
        end_name = sampled_connections[i]["end"]
        distance_real = sampled_connections[i]["distance"]
        distance_data_i = distance_data[i].item()
        distance_embeddings_i = distance_embeddings[i].item()
        connections_distances_data.append({
            "start": start_name,
            "end": end_name,
            "distance_real": distance_real,
            "distance_data": distance_data_i,
            "distance_embeddings": distance_embeddings_i
        })

    return connections_distances_data


def eval_distances_threshold_averages_raw_data(storage: StorageSuperset2,
                                               real_distance_threshold):
    connections_distances_data = _get_connection_distances_raw_data(storage)

    REAL_DISTANCE_THRESHOLD = real_distance_threshold
    average_distance_embeddings = 0
    average_distance_data = 0
    total_count = 0

    for connection in connections_distances_data:
        real_distance = connection["distance_real"]
        if real_distance < REAL_DISTANCE_THRESHOLD:
            total_count += 1
            average_distance_embeddings += connection["distance_embeddings"]
            average_distance_data += connection["distance_data"]

    if total_count == 0:
        print("No connections found")
        return 0, 0
    average_distance_embeddings /= total_count
    average_distance_data /= total_count

    print(f"Average distance embeddings: {average_distance_embeddings}")
    print(f"Average distance data: {average_distance_data}")
    return average_distance_embeddings, average_distance_data
