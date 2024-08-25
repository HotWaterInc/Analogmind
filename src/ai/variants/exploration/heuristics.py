import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.networks.adjacency_detector import AdjacencyDetector
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.ai.variants.exploration.others.neighborhood_network_thetas import NeighborhoodNetworkThetas
from src.ai.variants.exploration.params import MAX_DISTANCE, STEP_DISTANCE
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.save_load_handlers.data_handle import write_other_data_to_file, serialize_object_other, \
    deserialize_object_other
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_sample_image, \
    detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute, \
    detach_robot_rotate_continuous_absolute, detach_robot_forward_continuous, detach_robot_sample_image_inference
import threading
import torch
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
from src.modules.policies.testing_image_data import test_images_accuracy, process_webots_image_to_embedding, \
    squeeze_out_resnet_output
from src.modules.policies.utils_lib import webots_radians_to_normal, radians_to_degrees
import torch
from src.utils import get_device


def find_adjacency_heuristic_raw_data(storage: StorageSuperset2, datapoint: Dict[str, any],
                                      distance_data) -> List[str]:
    current_name = datapoint["name"]

    datapoints_names = storage.get_all_datapoints()
    adjacent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 1)
    adjacent_names.append(current_name)

    current_data_arr = []
    other_datapoints_data_arr = []
    selected_names = []

    for name in datapoints_names:
        if name in adjacent_names or name == current_name:
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
    adjacent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 1)
    adjacent_names.append(current_name)

    current_data_arr = []
    other_datapoints_data_arr = []
    selected_names = []

    for name in datapoints_names:
        if name in adjacent_names or name == current_name:
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
    adjacent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 1)
    adjacent_names.append(current_name)

    selected_names = []

    for name in datapoints_names:
        if name in adjacent_names or name == current_name:
            continue

        selected_names.append(name)

    if len(selected_names) == 0:
        return []

    length = len(selected_names)
    name_keys = {}

    found_additional_connections = []
    for i in range(length):
        name = selected_names[i]
        real_distance = storage.get_datapoints_real_distance(current_name, name)
        if real_distance < 0.65:
            found_additional_connections.append(name)

    return found_additional_connections
