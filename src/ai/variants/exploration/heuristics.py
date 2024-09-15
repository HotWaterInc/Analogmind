import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.ai.variants.exploration.networks.adjacency_detector import AdjacencyDetector
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.ai.variants.exploration.others.neighborhood_network_thetas import NeighborhoodNetworkThetas
from src.ai.variants.exploration.params import MAX_DISTANCE, STEP_DISTANCE
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai, load_custom_ai, load_other_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import *
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
import torch
from src.utils import get_device


def find_adjacency_heuristic_raw_data(storage: StorageSuperset2, datapoint: Dict[str, any],
                                      distance_data) -> List[str]:
    current_name = datapoint["name"]

    datapoints_names = storage.get_all_datapoints()

    current_data_arr = []
    other_datapoints_data_arr = []
    selected_names = []

    for name in datapoints_names:
        if name == current_name:
            continue

        for i in range(ROTATIONS):
            current_data = storage.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(current_name, i)
            existing_data = storage.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(name, i)

            current_data_arr.append(current_data)
            other_datapoints_data_arr.append(existing_data)
            selected_names.append(name)

    if len(current_data_arr) == 0:
        return []

    current_data_arr = torch.stack(current_data_arr).to(get_device())
    other_datapoints_data_arr = torch.stack(other_datapoints_data_arr).to(get_device())
    norm_distance = torch.norm(current_data_arr - other_datapoints_data_arr, p=2, dim=1)

    length = len(norm_distance)
    name_keys = {}

    found_additional_connections = []
    for i in range(length):
        distance_data_i = norm_distance[i]
        if distance_data_i.item() < distance_data:
            if selected_names[i] not in name_keys:
                name_keys[selected_names[i]] = 1
            else:
                name_keys[selected_names[i]] += 1

    for name in name_keys:
        if name_keys[name] >= 5:
            found_additional_connections.append(name)

    return found_additional_connections


def find_adjacency_heuristic_adjacency_network(storage: StorageSuperset2, datapoint: Dict[str, any],
                                               adjacency_network: AdjacencyDetector) -> List[str]:
    current_name = datapoint["name"]

    datapoints_names = storage.get_all_datapoints()

    current_data_arr = []
    other_datapoints_data_arr = []
    selected_names = []

    for name in datapoints_names:
        if name == current_name:
            continue

        for i in range(ROTATIONS):
            current_data = storage.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(current_name, i)
            existing_data = storage.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(name, i)

            current_data_arr.append(current_data)
            other_datapoints_data_arr.append(existing_data)
            selected_names.append(name)

    if len(current_data_arr) == 0:
        return []

    current_data_arr = torch.stack(current_data_arr).to(get_device())
    other_datapoints_data_arr = torch.stack(other_datapoints_data_arr).to(get_device())
    probabilities = adjacency_network(current_data_arr, other_datapoints_data_arr)

    length = len(selected_names)
    name_keys = {}

    found_additional_connections = []
    threshold_true = 0.65

    # print("Probabilities", probabilities[:10])

    for i in range(length):
        probability_sample = probabilities[i]
        if probability_sample[0] > threshold_true:
            if selected_names[i] not in name_keys:
                name_keys[selected_names[i]] = 1
            else:
                name_keys[selected_names[i]] += 1

    for name in name_keys:
        if name_keys[name] >= 12:
            found_additional_connections.append(name)

    return found_additional_connections


def find_adjacency_heuristic_cheating(storage: StorageSuperset2, datapoint: Dict[str, any]) -> List[str]:
    current_name = datapoint["name"]

    datapoints_names = storage.get_all_datapoints()

    selected_names = []

    for name in datapoints_names:
        # other conditions
        if name == current_name:
            continue
        selected_names.append(name)

    if len(selected_names) == 0:
        return []

    length = len(selected_names)

    found_additional_connections = []
    for i in range(length):
        name = selected_names[i]
        real_distance = storage.get_datapoints_real_distance(current_name, name)
        if real_distance < STEP_DISTANCE_CLOSE_THRESHOLD:
            found_additional_connections.append(name)

    return found_additional_connections
