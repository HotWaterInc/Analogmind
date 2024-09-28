from typing import Dict, List

from src.navigation_core.autonomous_exploration.metrics.heuristics import \
    find_adjacency_heuristic_adjacency_network, \
    find_adjacency_heuristic_raw_data, find_adjacency_heuristic_by_metadata
from src.navigation_core.autonomous_exploration.networks.adjacency_detector.network_class import AdjacencyDetector
from src.runtime_storages.storage_struct import StorageStruct
from src.runtime_storages.types import NodeAuthenticData


def build_find_adjacency_heuristic_cheating():
    def find_adjacency_heuristic_augmented(storage: StorageStruct, datapoint: NodeAuthenticData):
        return find_adjacency_heuristic_by_metadata(storage, datapoint)

    return find_adjacency_heuristic_augmented


def build_find_adjacency_heursitic_adjacency_network(adjacency_network: AdjacencyDetector):
    def find_adjacency_heursitic_augmented(storage: StorageStruct, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_adjacency_network(storage, datapoint, adjacency_network)

    return find_adjacency_heursitic_augmented

# def build_find_adjacency_heursitic_raw_data(storage: StorageStruct):
#     distance_embeddings, distance_data = eval_distances_threshold_averages_raw_data(storage,
#                                                                                     real_distance_threshold=0.5)
#     distance_data *= 0.65
#
#     def find_adjacency_heursitic_augmented(storage: StorageStruct, datapoint: Dict[str, any]):
#         return find_adjacency_heuristic_raw_data(storage, datapoint, distance_data)
#
#     return find_adjacency_heursitic_augmented
