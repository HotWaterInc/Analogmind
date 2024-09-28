import time
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Union

from src.navigation_core.autonomous_exploration.networks.adjacency_detector.network_class import AdjacencyDetector
from src.navigation_core.to_refactor.params import ROTATIONS
from src.navigation_core.autonomous_exploration.params import IS_CLOSE_THRESHOLD
from src.runtime_storages.functions.pure_functions import eulerian_distance
from src.runtime_storages.storage_struct import StorageStruct
from src.runtime_storages.types import NodeAuthenticData
import torch
from src import runtime_storages as storage


def find_adjacency_heuristic_raw_data(storage_struct: StorageStruct, datapoint: Dict[str, any],
                                      distance_data) -> List[str]:
    pass
    # current_name = datapoint["name"]
    #
    # datapoints_names = storage_struct.get_all_datapoints()
    #
    # current_data_arr = []
    # other_datapoints_data_arr = []
    # selected_names = []
    #
    # for name in datapoints_names:
    #     if name == current_name:
    #         continue
    #
    #     for i in range(ROTATIONS):
    #         current_data = storage_struct.node_get_datapoint_tensor_at_index_noisy(current_name, i)
    #         existing_data = storage_struct.node_get_datapoint_tensor_at_index_noisy(name, i)
    #
    #         current_data_arr.append(current_data)
    #         other_datapoints_data_arr.append(existing_data)
    #         selected_names.append(name)
    #
    # if len(current_data_arr) == 0:
    #     return []
    #
    # current_data_arr = torch.stack(current_data_arr).to(get_device())
    # other_datapoints_data_arr = torch.stack(other_datapoints_data_arr).to(get_device())
    # norm_distance = torch.norm(current_data_arr - other_datapoints_data_arr, p=2, dim=1)
    #
    # length = len(norm_distance)
    # name_keys = {}
    #
    # found_additional_connections = []
    # for i in range(length):
    #     distance_data_i = norm_distance[i]
    #     if distance_data_i.item() < distance_data:
    #         if selected_names[i] not in name_keys:
    #             name_keys[selected_names[i]] = 1
    #         else:
    #             name_keys[selected_names[i]] += 1
    #
    # for name in name_keys:
    #     if name_keys[name] >= 5:
    #         found_additional_connections.append(name)
    #
    # return found_additional_connections


def find_adjacency_heuristic_adjacency_network(storage_struct: StorageStruct, datapoint: Dict[str, any],
                                               adjacency_network: AdjacencyDetector) -> List[str]:
    pass
    # current_name = datapoint["name"]
    #
    # datapoints_names = storage_struct.get_all_datapoints()
    #
    # current_data_arr = []
    # other_datapoints_data_arr = []
    # selected_names = []
    #
    # for name in datapoints_names:
    #     if name == current_name:
    #         continue
    #
    #     for i in range(ROTATIONS):
    #         current_data = storage_struct.node_get_datapoint_tensor_at_index_noisy(current_name, i)
    #         existing_data = storage_struct.node_get_datapoint_tensor_at_index_noisy(name, i)
    #
    #         current_data_arr.append(current_data)
    #         other_datapoints_data_arr.append(existing_data)
    #         selected_names.append(name)
    #
    # if len(current_data_arr) == 0:
    #     return []
    #
    # current_data_arr = torch.stack(current_data_arr).to(get_device())
    # other_datapoints_data_arr = torch.stack(other_datapoints_data_arr).to(get_device())
    # probabilities = adjacency_network(current_data_arr, other_datapoints_data_arr)
    #
    # length = len(selected_names)
    # name_keys = {}
    #
    # found_additional_connections = []
    # threshold_true = 0.65
    #
    # # print("Probabilities", probabilities[:10])
    #
    # for i in range(length):
    #     probability_sample = probabilities[i]
    #     if probability_sample[0] > threshold_true:
    #         if selected_names[i] not in name_keys:
    #             name_keys[selected_names[i]] = 1
    #         else:
    #             name_keys[selected_names[i]] += 1
    #
    # for name in name_keys:
    #     if name_keys[name] >= 12:
    #         found_additional_connections.append(name)
    #
    # return found_additional_connections


def find_adjacency_heuristic_by_metadata(storage_struct: StorageStruct, current_node: NodeAuthenticData) -> List[str]:
    current_name = current_node["name"]

    nodes_names = storage.nodes_get_all_names(storage_struct)
    selected_names = []

    for name in nodes_names:
        if name == current_name:
            continue
        selected_names.append(name)

    if len(selected_names) == 0:
        return []

    length = len(selected_names)
    found_additional_connections = []
    for i in range(length):
        name = selected_names[i]

        real_distance = storage.get_distance_between_nodes_metadata(storage_struct, current_name, name)
        if real_distance < IS_CLOSE_THRESHOLD:
            found_additional_connections.append(name)

    return found_additional_connections
