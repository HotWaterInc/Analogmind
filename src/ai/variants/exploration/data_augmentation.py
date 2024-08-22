from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric
from src.ai.runtime_data_storage.storage_superset2 import *
from src.ai.variants.exploration.metric_builders import build_find_adjacency_heursitic_adjacency_network, \
    build_find_adjacency_heursitic_raw_data
from src.utils import get_device


def augment_data_network_heuristic(storage: StorageSuperset2, adjacency_network) -> any:
    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    adjacency_network = adjacency_network.to(get_device())
    adjacency_network.eval()

    distance_metric = build_find_adjacency_heursitic_adjacency_network(adjacency_network)
    new_connections = evaluate_distance_metric(storage, distance_metric, storage.get_all_datapoints())
    return new_connections


def load_storage_with_base_data(storage: StorageSuperset2):
    random_walk_datapoints = []
    random_walk_connections = []

    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    return storage


def augment_storage_with_saved_connections(storage: StorageSuperset2) -> any:
    new_connections1 = read_other_data_from_file(f"additional_found_connections_rawh_random_walks_300.json")
    flag_data_authenticity(new_connections1)
    storage.incorporate_new_data([], new_connections1)
    new_connections2 = read_other_data_from_file(f"additional_found_connections_networkh_random_walks_300_24rot.json")
    flag_data_authenticity(new_connections2)
    storage.incorporate_new_data([], new_connections2)
    return storage


def augment_data_raw_heuristic(storage: StorageSuperset2) -> any:
    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    distance_metric = build_find_adjacency_heursitic_raw_data(storage)
    new_connections = evaluate_distance_metric(storage, distance_metric, storage.get_all_datapoints())
    return new_connections
