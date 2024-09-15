import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.ai.variants.exploration.heuristics import find_adjacency_heuristic_raw_data, \
    find_adjacency_heuristic_adjacency_network, find_adjacency_heuristic_cheating
from src.ai.variants.exploration.networks.adjacency_detector import AdjacencyDetector
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric, \
    eval_distances_threshold_averages_raw_data
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.ai.variants.exploration.others.neighborhood_network_thetas import NeighborhoodNetworkThetas
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import *
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
import torch
from src.utils import get_device


def build_find_adjacency_heursitic_cheating():
    def find_adjacency_heursitic_augmented(storage: StorageSuperset2, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_cheating(storage, datapoint)

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
