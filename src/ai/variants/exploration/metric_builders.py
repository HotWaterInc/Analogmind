import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.heuristics import find_adjacency_heuristic_neighborhood_network_thetas, \
    find_adjacency_heuristic_raw_data, find_adjacency_heuristic_adjacency_network
from src.ai.variants.exploration.networks.adjacency_detector import AdjacencyDetector
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric, \
    eval_distances_threshold_averages_raw_data
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.ai.variants.exploration.others.neighborhood_network_thetas import NeighborhoodNetworkThetas
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


def build_find_adjacency_heursitic_neighborhood_network_thetas(
        neighborhood_network: NeighborhoodNetworkThetas):
    def find_adjacency_heursitic_augmented(storage: StorageSuperset2, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_neighborhood_network_thetas(storage, neighborhood_network, datapoint)

    return find_adjacency_heursitic_augmented


def build_find_adjacency_heursitic_adjacency_network(adjacency_network: AdjacencyDetector):
    def find_adjacency_heursitic_augmented(storage: StorageSuperset2, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_adjacency_network(storage, datapoint, adjacency_network)

    return find_adjacency_heursitic_augmented


def build_find_adjacency_heursitic_raw_data(storage: StorageSuperset2):
    distance_embeddings, distance_data = eval_distances_threshold_averages_raw_data(storage,
                                                                                    real_distance_threshold=0.5)
    distance_data *= 0.65

    def find_adjacency_heursitic_augmented(storage: StorageSuperset2, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_raw_data(storage, datapoint, distance_data)

    return find_adjacency_heursitic_augmented


def build_augmented_connections(storage: StorageSuperset2, distance_metric, new_datapoints: List[any]):
    additional_connections = []
    for idx, new_datapoint in enumerate(new_datapoints):
        # run distance metric
        found_adjacent: List[str] = distance_metric(storage, new_datapoint)

        # add connections
        additional_connections.append({
            "start": new_datapoint["name"],
            "end": found_adjacent,
            "distance": None,
            "direction": None
        })

    return additional_connections
