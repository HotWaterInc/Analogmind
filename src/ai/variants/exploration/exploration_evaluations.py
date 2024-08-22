import numpy as np
import torch
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.utils import array_to_tensor, get_device
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from typing import List
from src.modules.time_profiler import start_profiler, profiler_checkpoint, profiler_checkpoint_blank
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display
from src.ai.variants.exploration.utils import check_min_distance
import time
import math
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.pretty_display import pretty_display_start, set_pretty_display, pretty_display
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
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
import torch
from src.utils import get_device


def ground_truth_metric(storage: StorageSuperset2, new_datapoint: Dict[str, any], lower_bound_distance_threshold,
                        upper_bound_distance_threshold):
    """
    Find the closest datapoint in the storage
    """
    current_name = new_datapoint["name"]
    datapoints_names = storage.get_all_datapoints()
    adjacent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 1)
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

    set_pretty_display(len(new_datapoints))
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

        true_datapoints_lower_end = ground_truth_metric(storage, new_datapoint, 0, 0.5)
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
    print("Percent of true positive of actual positives", true_positive / len(true_found_datapoints))
    print("True positive distant", true_positives_distant)
    print("Avg distance of true positive distant", avg_distant_true_positive / true_positives_distant)
    print("Really bad false positive", really_bad_false_positive)
    print("")
    # print("False positive distances", false_positive_distances)


def evaluate_distance_metric(storage: StorageSuperset2, metric, new_datapoints: List[any]
                             ):
    """
    Evaluate new datapoints and old datapoints with the distance metric
    """
    true_positive = 0
    true_positives_distant = 0
    really_bad_false_positive = 0

    # new_datapoints = new_datapoints[:50]

    set_pretty_display(len(new_datapoints))
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

        true_datapoints_lower_end = ground_truth_metric(storage, new_datapoint, 0, 0.5)
        true_found_datapoints.extend(true_datapoints_lower_end)

        for founddp in found_datapoints:
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

    print("")
    print("True positive", true_positive)
    print("Percent of true positive of actual positives", true_positive / len(true_found_datapoints))
    print("True positive distant", true_positives_distant)
    print("Avg distance of true positive distant", avg_distant_true_positive / true_positives_distant)
    print("Really bad false positive", really_bad_false_positive)
    print("")
    # print("False positive distances", false_positive_distances)

    return new_connections_pairs


def _get_connection_distances_seen_network(storage: StorageSuperset2, seen_network: SeenNetwork) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(100, len(connections))
    seen_network.eval()
    seen_network = seen_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
        end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)

        start_data_arr.append(start_data)
        end_data_arr.append(end_data)

    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())

    start_embedding = seen_network.encoder_inference(start_data_arr)
    end_embedding = seen_network.encoder_inference(end_data_arr)

    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distance_embeddings = torch.norm(start_embedding - end_embedding, p=2, dim=1)

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


def print_distances_embeddings_inputs(storage: StorageSuperset2, seen_network: SeenNetwork):
    """
    Evaluates the relationship between the distances between the embeddings and the inputs
    """

    connections_distances_data = _get_connection_distances_seen_network(storage, seen_network)
    # sort by real distance
    connections_distances_data.sort(key=lambda x: x["distance_real"])
    for connection in connections_distances_data:
        print(f"{connection['distance_real']} => {connection['distance_embeddings']} || {connection['distance_data']}")


def eval_distances_threshold_averages_raw_data(storage: StorageSuperset2,
                                               real_distance_threshold):
    connections_distances_data = _get_connection_distances_raw_data_north(storage)

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


def eval_distances_threshold_averages_seen_network(storage: StorageSuperset2, seen_network: SeenNetwork,
                                                   real_distance_threshold):
    connections_distances_data = _get_connection_distances_seen_network(storage, seen_network)

    REAL_DISTANCE_THRESHOLD = real_distance_threshold
    average_distance_embeddings = 0
    average_distance_data = 0
    total_count = 0

    print("Connections distances data: ", connections_distances_data)

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
