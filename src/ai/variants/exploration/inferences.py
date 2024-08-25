import random
import torch
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, generate_connection, \
    direction_thetas_to_radians
from src.ai.variants.exploration.inference_policy import generate_dxdy
from src.ai.variants.exploration.networks.images_raw_distance_predictor import ImagesRawDistancePredictor
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork
from typing import List
from src.ai.variants.exploration.params import ROTATIONS
from src.utils import get_device
from torch import nn


def fill_augmented_connections_directions(additional_connections: List[any], storage: StorageSuperset2,
                                          image_direction_network: nn.Module):
    additional_connections_augmented = []

    start_data_array = []
    end_data_array = []

    start_name_array = []
    end_name_array = []
    distances_array = []
    index_array = []

    directions_thetas_hashmap = {}
    SELECTIONS = 4
    for idx, connection in enumerate(additional_connections):
        start = connection["start"]
        end = connection["end"]

        for i in range(SELECTIONS):
            i = random.randint(0, ROTATIONS - 1)
            start_data = storage.get_datapoint_data_tensor_by_name(start)[i]
            end_data = storage.get_datapoint_data_tensor_by_name(end)[i]

            start_data_array.append(start_data)
            end_data_array.append(end_data)
            index_array.append(idx)

    start_data_array = torch.stack(start_data_array).to(get_device())
    end_data_array = torch.stack(end_data_array).to(get_device())
    direction_thetas = image_direction_network(start_data_array, end_data_array)

    for idx, direction_thetas in enumerate(direction_thetas):
        index_connection = index_array[idx]
        if index_connection not in directions_thetas_hashmap:
            directions_thetas_hashmap[index_connection] = torch.tensor(0.0, device=get_device())
        directions_thetas_hashmap[index_connection] += direction_thetas

    for idx in directions_thetas_hashmap:
        start_name = additional_connections[idx]["start"]
        end_name = additional_connections[idx]["end"]
        direction_thetas = directions_thetas_hashmap[idx] / SELECTIONS
        direction = direction_thetas_to_radians(direction_thetas)

        distance_authenticity = additional_connections[idx]["markings"]["distance"] == "authentic"
        distance = additional_connections[idx]["distance"]
        direction = generate_dxdy(direction, distance)

        new_connections = generate_connection(
            start=start_name,
            end=end_name,
            distance=distance,
            direction=direction,
            distance_authenticity=distance_authenticity,
            direction_authenticity=direction
        )
        additional_connections_augmented.append(new_connections)

    return additional_connections_augmented


def fill_augmented_connections_directions_cheating(additional_connections: List[any], storage: StorageSuperset2):
    additional_connections_augmented = []

    for idx, connection in enumerate(additional_connections):
        start = connection["start"]
        end = connection["end"]
        dist = connection["distance"]
        direction = storage.get_datapoints_real_direction(start, end)
        distance_authenticity = connection["markings"]["distance"] == "authentic"
        direction_authenticity = connection["markings"]["direction"] == "authentic"

        additional_connections_augmented.append(
            generate_connection(
                start=start,
                end=end,
                distance=dist,
                direction=direction,
                distance_authenticity=distance_authenticity,
                direction_authenticity=direction_authenticity
            )
        )

    return additional_connections_augmented


def fill_augmented_connections_distances_cheating(additional_connections: List[any], storage: StorageSuperset2):
    additional_connections_augmented = []

    for idx, connection in enumerate(additional_connections):
        start = connection["start"]
        end = connection["end"]
        dist = storage.get_datapoints_real_distance(start, end)
        direction = connection["direction"]
        distance_authenticity = connection["markings"]["distance"] == "authentic"
        direction_authenticity = connection["markings"]["direction"] == "authentic"

        additional_connections_augmented.append(
            generate_connection(
                start=start,
                end=end,
                distance=dist,
                direction=direction,
                distance_authenticity=distance_authenticity,
                direction_authenticity=direction_authenticity
            )
        )

    return additional_connections_augmented


def fill_augmented_connections_distances(additional_connections: List[any], storage: StorageSuperset2,
                                         image_distance_network: ImagesRawDistancePredictor):
    additional_connections_augmented = []

    start_data_array = []
    end_data_array = []

    start_name_array = []
    end_name_array = []
    directions_array = []
    index_array = []

    distances_hashmap = {}
    SELECTIONS = 4
    for idx, connection in enumerate(additional_connections):
        start = connection["start"]
        end = connection["end"]

        for i in range(SELECTIONS):
            i = random.randint(0, ROTATIONS - 1)
            start_data = storage.get_datapoint_data_tensor_by_name(start)[i]
            end_data = storage.get_datapoint_data_tensor_by_name(end)[i]

            start_data_array.append(start_data)
            end_data_array.append(end_data)
            index_array.append(idx)

    if len(start_data_array) == 0:
        return additional_connections_augmented

    start_data_array = torch.stack(start_data_array).to(get_device())
    end_data_array = torch.stack(end_data_array).to(get_device())
    distances = image_distance_network(start_data_array, end_data_array)

    for idx, dist in enumerate(distances):
        index_connection = index_array[idx]
        if index_connection not in distances_hashmap:
            distances_hashmap[index_connection] = 0
        distances_hashmap[index_connection] += dist.item()

    for idx in distances_hashmap:
        start_name = additional_connections[idx]["start"]
        end_name = additional_connections[idx]["end"]
        distance = distances_hashmap[idx] / SELECTIONS
        direction = additional_connections[idx]["direction"]
        direction_authenticity = additional_connections[idx]["markings"]["direction"] == "authentic"

        new_connections = generate_connection(
            start=start_name,
            end=end_name,
            distance=distance,
            direction=direction,
            distance_authenticity=False,  # because it is generated
            direction_authenticity=direction_authenticity
        )
        additional_connections_augmented.append(new_connections)

    return additional_connections_augmented
