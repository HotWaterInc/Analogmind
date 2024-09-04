from typing import Generator
from src.ai.variants.exploration.metric_builders import build_find_adjacency_heursitic_raw_data, \
    build_find_adjacency_heursitic_adjacency_network
from src.ai.variants.exploration.networks.SDirDistState_network import SDirDistState
from src.ai.variants.exploration.networks.SSDir_network import SSDirNetwork
from src.ai.variants.exploration.networks.adjacency_detector import AdjacencyDetector, \
    train_adjacency_network_until_threshold
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork
from src.ai.variants.exploration.params import STEP_DISTANCE, ROTATIONS, MAX_DISTANCE
from src.modules.pretty_display import pretty_display_start, pretty_display_set_and_start, pretty_display
from src.modules.save_load_handlers.data_handle import write_other_data_to_file
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_teleport_absolute, \
    detach_robot_sample_image_inference
import time
import torch.nn as nn
from src.ai.runtime_data_storage.storage_superset2 import *
from typing import List, Dict
from src.utils import get_device
from src.ai.variants.exploration.networks.manifold_network import ManifoldNetwork
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric, \
    evaluate_distance_metric_on_already_found_connections


def augment_data_testing_network_distance(storage: StorageSuperset2,
                                          distance_raw) -> any:
    distance_raw = distance_raw.to(get_device())
    distance_raw.eval()

    new_datapoints = storage.get_all_datapoints()
    pretty_display_set_and_start(len(new_datapoints))

    def embedding_emtpy_policy(x):
        return x

    def embedding_policy(x):
        x_distances = []
        for thetas in x:
            dist = distance_thetas_to_distance_percent(thetas) * MAX_DISTANCE
            x_distances.append(dist)

        return torch.tensor(x_distances).to(get_device())

    new_datapoints = new_datapoints[:25]
    average_loss_with_raw = 0
    for idx, new_datapoint in enumerate(new_datapoints):
        print("Processing", idx, "out of", len(new_datapoints))
        new_datapoint = storage.get_datapoint_by_name(new_datapoint)
        average_loss_with_raw += find_adjacency_heuristic_distance_thetas(storage, new_datapoint, distance_raw,
                                                                          embedding_emtpy_policy)
    average_loss_with_thetas = 0
    # for idx, new_datapoint in enumerate(new_datapoints):
    #     print("Processing", idx, "out of", len(new_datapoints))
    #     new_datapoint = storage.get_datapoint_by_name(new_datapoint)
    #     average_loss_with_thetas += find_adjacency_heuristic_distance_thetas(storage, new_datapoint, distance_thetas,
    #                                                                          embedding_policy)

    average_loss_with_thetas /= len(new_datapoints)
    average_loss_with_raw /= len(new_datapoints)

    print("Average loss with raw", average_loss_with_raw)


def augment_storage_connections_with_synthetic_distances(storage: StorageSuperset2, distance_network: nn.Module):
    """
    Augments all synthetic or non-authentic connections with distances
    """
    distance_network = distance_network.to(get_device())
    distance_network.eval()

    storage.get_all_connections_only_datapoints_authenticity_filter()

    pass


def find_adjacency_heuristic_distance_thetas(storage: StorageSuperset2, datapoint: Dict[str, any],
                                             distance_thetas: nn.Module, embedding_policy):
    current_datapoint_name = datapoint["name"]

    datapoints_names = storage.get_all_datapoints()
    adjacent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(current_datapoint_name, 1)
    adjacent_names.append(current_datapoint_name)

    current_data_arr = []
    other_datapoints_data_arr = []
    selected_names = []
    SELECTIONS = 4

    for name in datapoints_names:
        if name in adjacent_names or name == current_datapoint_name:
            continue

        real_distance = get_real_distance_between_datapoints(storage.get_datapoint_by_name(current_datapoint_name),
                                                             storage.get_datapoint_by_name(name))
        if real_distance > MAX_DISTANCE - 2:
            continue

        for i in range(SELECTIONS):
            i = random.randint(0, ROTATIONS - 1)
            current_data = storage.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(
                current_datapoint_name, i)
            existing_data = storage.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(name, i)

            current_data_arr.append(current_data)
            other_datapoints_data_arr.append(existing_data)
            selected_names.append(name)

    current_data_arr = torch.stack(current_data_arr).to(get_device())
    other_datapoints_data_arr = torch.stack(other_datapoints_data_arr).to(get_device())
    lengths = distance_thetas(current_data_arr, other_datapoints_data_arr)
    lengths = embedding_policy(lengths)

    array_length = len(selected_names)
    name_keys = {}

    for i in range(array_length):
        length = lengths[i]
        if selected_names[i] not in name_keys:
            name_keys[selected_names[i]] = length
        else:
            name_keys[selected_names[i]] += length

    for name in name_keys:
        name_keys[name] /= SELECTIONS

    real_distances = []
    predicted_distances = []
    for name in name_keys:
        datapoint1 = storage.get_datapoint_by_name(current_datapoint_name)
        datapoint2 = storage.get_datapoint_by_name(name)
        real_distance = get_real_distance_between_datapoints(datapoint1, datapoint2)
        predicted_distance = name_keys[name]

        real_distances.append(real_distance)
        predicted_distances.append(predicted_distance)

        # print("Real distance", real_distance, "Predicted distance", predicted_distance.item())

    real_distances = torch.tensor(real_distances).to(get_device())
    predicted_distances = torch.tensor(predicted_distances).to(get_device())
    criterion = nn.MSELoss()
    loss = criterion(predicted_distances, real_distances)

    # print("Loss", loss.item())
    return loss.item()
