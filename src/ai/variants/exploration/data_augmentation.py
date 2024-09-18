from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric
from src.ai.runtime_storages.storage_superset2 import *
from src.ai.variants.exploration.metric_builders import build_find_adjacency_heursitic_adjacency_network, \
    build_find_adjacency_heursitic_raw_data
from src.ai.variants.exploration.utils_pure_functions import flag_data_authenticity
from src.utils import get_device
from torch import nn


def data_augmentation_network_heuristic(storage: StorageSuperset2, adjacency_network) -> any:
    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    adjacency_network = adjacency_network.to(get_device())
    adjacency_network.eval()

    distance_metric = build_find_adjacency_heursitic_adjacency_network(adjacency_network)
    new_connections = evaluate_distance_metric(storage, distance_metric, storage.get_all_datapoints())
    return new_connections


def load_storage_with_base_data(storage: StorageSuperset2, datapoints_filename: str, connections_filename: str) -> any:
    random_walk_datapoints = read_other_data_from_file(datapoints_filename)
    random_walk_connections = read_other_data_from_file(connections_filename)
    flag_data_authenticity(random_walk_connections)
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    return storage


def storage_augment_with_saved_augmented_connections(storage: StorageSuperset2) -> any:
    new_connections1_augmented = read_other_data_from_file(
        f"additional_found_connections_rawh_random_walks_300_distance_augmented.json")

    for idx, conn in enumerate(new_connections1_augmented):
        conn["markings"] = get_markings(False, False)

    storage.incorporate_new_data([], new_connections1_augmented)
    new_connections2_augmented = read_other_data_from_file(
        f"additional_found_connections_networkh_random_walks_300_distance_augmented.json")

    for idx, conn in enumerate(new_connections2_augmented):
        conn["markings"] = get_markings(False, False)

    storage.incorporate_new_data([], new_connections2_augmented)
    return storage


def get_augmented_connections() -> any:
    new_connections1 = read_other_data_from_file(f"additional_found_connections_rawh_random_walks_300.json")
    flag_data_authenticity(new_connections1)
    new_connections2 = read_other_data_from_file(f"additional_found_connections_networkh_random_walks_300.json")
    flag_data_authenticity(new_connections2)
    return new_connections1 + new_connections2


def storage_augment_with_saved_connections_already_augmented(storage: StorageSuperset2) -> any:
    new_connections1 = read_other_data_from_file(
        f"additional_found_connections_rawh_random_walks_300_distance_augmented.json")
    flag_data_authenticity(new_connections1)
    storage.incorporate_new_data([], new_connections1)
    new_connections2 = read_other_data_from_file(
        f"additional_found_connections_networkh_random_walks_300_distance_augmented.json")
    flag_data_authenticity(new_connections2)
    storage.incorporate_new_data([], new_connections2)
    return storage


def storage_augment_with_saved_connections(storage: StorageSuperset2) -> any:
    new_connections1 = read_other_data_from_file(f"additional_found_connections_rawh_random_walks_300.json")
    flag_data_authenticity(new_connections1)
    storage.incorporate_new_data([], new_connections1)
    new_connections2 = read_other_data_from_file(f"additional_found_connections_networkh_random_walks_300.json")
    flag_data_authenticity(new_connections2)
    storage.incorporate_new_data([], new_connections2)
    return storage


def augment_saved_connections_with_distances(storage: StorageSuperset2, distance_network: nn.Module, filename: str):
    connections = read_other_data_from_file(filename)
    data_augment_connection_distances(storage, distance_network, connections, debug=True)
    return connections


def data_augment_raw_heuristic(storage: StorageSuperset2) -> any:
    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    distance_metric = build_find_adjacency_heursitic_raw_data(storage)
    new_connections = evaluate_distance_metric(storage, distance_metric, storage.get_all_datapoints())
    return new_connections


def data_augment_connection_distances(storage: StorageSuperset2, distance_network, connections, debug: bool = False):
    """
    Augments all synthetic or non-authentic connections with distances
    """
    distance_network = distance_network.to(get_device())
    distance_network.eval()

    SELECTIONS = 6
    err = 0

    start_data_arr = []
    end_data_arr = []
    real_distances = []  # for debugging
    indexes = []

    for idx, connection in enumerate(connections):
        if connection["distance"] is not None:
            continue

        start = connection["start"]
        end = connection["end"]
        direction = connection["direction"]
        distance = connection["distance"]
        real_distance = None

        for i in range(SELECTIONS):
            i = random.randint(0, ROTATIONS - 1)
            start_data = storage.node_get_datapoint_tensor_at_index_noisy(start, i)
            end_data = storage.node_get_datapoint_tensor_at_index_noisy(end, i)
            index = idx

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            indexes.append(index)

    start_data_tensor = torch.stack(start_data_arr).to(get_device())
    end_data_tensor = torch.stack(end_data_arr).to(get_device())
    synthetic_distances = distance_network(start_data_tensor, end_data_tensor)
    synthetic_distances = synthetic_distances.squeeze().tolist()
    synthetic_distances_hashmap = {}

    for idx, synthetic_distance in enumerate(synthetic_distances):
        index = indexes[idx]
        if index not in synthetic_distances_hashmap:
            synthetic_distances_hashmap[index] = 0

        synthetic_distances_hashmap[index] += synthetic_distance

    pred_dist = []
    for hash_index in synthetic_distances_hashmap:
        synthetic_distances_hashmap[hash_index] /= SELECTIONS

        connection = connections[hash_index]
        connection["distance"] = synthetic_distances_hashmap[hash_index]
        pred_dist.append(synthetic_distances_hashmap[hash_index])

        if debug:
            real_distance = get_real_distance_between_datapoints(storage.node_get_by_name(connection["start"]),
                                                                 storage.node_get_by_name(connection["end"]))
            real_distances.append(real_distance)

    if debug:
        # debugging purposes again
        pred_dist = torch.tensor(pred_dist)
        real_distances = torch.tensor(real_distances)
        mse_loss = torch.nn.MSELoss()
        err = mse_loss(pred_dist, real_distances)
        print("Error for synthetically generated distances", err.item())
