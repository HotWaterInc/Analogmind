import torch
import numpy as np
import math
from typing import List
from src.navigation_core.autonomous_exploration.types import ConnectionFrontier
from src.navigation_core.pure_functions import direction_to_degrees_atan, degrees_to_percent, \
    angle_percent_to_thetas_normalized_cached, find_thetas_null_indexes, get_angle_percent_from_thetas_index, \
    angle_percent_to_radians, generate_dxdy
from src.navigation_core.to_refactor.params import STEP_DISTANCE, DIRECTION_REPRESENTATION_SIZE
from src.runtime_storages.storage_struct import StorageStruct
from src import runtime_storages as storage


def get_missing_connections_based_on_distance(storage_struct: StorageStruct, datapoint, distance_threshold):
    datapoints_names = storage_struct.get_all_datapoints()
    current_x = datapoint["params"]["x"]
    current_y = datapoint["params"]["y"]
    current_name = datapoint["name"]
    adjcent_names = storage_struct.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(current_name, 2)
    adjcent_names.append(current_name)

    found_connections = []

    for name in datapoints_names:
        if name in adjcent_names or name == current_name:
            continue

        data = storage_struct.node_get_by_name(name)

        data_x = data["params"]["x"]
        data_y = data["params"]["y"]
        data_name = name

        real_distance = np.sqrt((current_x - data_x) ** 2 + (current_y - data_y) ** 2)
        if real_distance < distance_threshold:
            found_connections.append({
                "start": current_name,
                "end": data_name,
                "distance": real_distance,
                "direction": None
            })

    return found_connections


def adjust_distance_sensors_according_to_rotation_duplicate(distance_sensors, rotation_percent):
    sensors_count = len(distance_sensors)
    rotation_percentage = rotation_percent
    rotation_offset = int(rotation_percentage * sensors_count)
    new_distance_sensors = np.zeros(sensors_count)

    for i in range(sensors_count):
        new_index = (i + rotation_offset) % sensors_count
        # new_distance_sensors.append(distance_sensors[new_index])
        new_distance_sensors[new_index] = distance_sensors[i]

    return new_distance_sensors


def adjust_distance_sensors_according_to_rotation(distance_sensors, rotation_percent):
    sensors_count = len(distance_sensors)
    rotation_percentage = rotation_percent
    rotation_offset = int(rotation_percentage * sensors_count)
    new_distance_sensors = np.zeros(sensors_count)

    for i in range(sensors_count):
        new_index = (i + rotation_offset) % sensors_count
        # new_distance_sensors.append(distance_sensors[new_index])
        new_distance_sensors[new_index] = distance_sensors[i]

    return new_distance_sensors


def data_to_random_manifold(data, index: int):
    new_data = torch.zeros(data.shape[0], MANIFOLD_SIZE)
    random_manifold = torch.rand(MANIFOLD_SIZE)

    for idx, val in enumerate(data):
        new_data[idx] = random_manifold

    return new_data


def check_datapoint_connections_completeness(storage_struct: StorageStruct, datapoint_name: str):
    direction_thetas_accumulated = np.zeros(DIRECTION_REPRESENTATION_SIZE)
    connections = storage_struct.get_datapoint_adjacent_connections_direction_filled(datapoint_name)
    connections_non_null = [connection for connection in connections if connection["end"] is not None]
    connections = connections_non_null

    for connection in connections:
        direction = connection["direction"]
        degrees = direction_to_degrees_atan(direction)
        angle_percent = degrees_to_percent(degrees)
        dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_REPRESENTATION_SIZE)

        np.add(direction_thetas_accumulated, dir_thetas, out=direction_thetas_accumulated)

    has_zeros = False
    for theta in direction_thetas_accumulated:
        if theta == 0:
            has_zeros = True
            break

    connection_completeness = not has_zeros and len(connections) >= 5

    return connection_completeness


def frontier_find_all_datapoints_and_directions(storage_struct: StorageStruct, return_first: bool = False,
                                                starting_point=None) -> List[
                                                                            ConnectionFrontier] | ConnectionFrontier | None:
    """
    Finds empty connections gaps and goes there to explore, starting from the current datapoint
    If no gaps are found it means the entire map was explored
    """

    current_datapoint_name = None
    if starting_point is not None:
        current_datapoint_name = starting_point
    else:
        current_datapoint_name = storage.nodes_get_all_names(storage_struct)[0]

    queue = [current_datapoint_name]
    visited = set()
    visited.add(current_datapoint_name)

    possible_frontiers = []

    def add_frontier(datapoint_name, direction) -> None:
        distance = STEP_DISTANCE
        frontier = ConnectionFrontier(
            start=datapoint_name,
            distance=distance,
            direction=direction
        )
        possible_frontiers.append(frontier)

    while queue:
        current_datapoint_name = queue.pop(0)

        connections = storage.node_get_connections_adjacent(storage_struct, current_datapoint_name)
        connections_null = storage.node_get_connections_null(storage_struct, current_datapoint_name)

        accumulated_thetas = np.zeros(DIRECTION_REPRESENTATION_SIZE)

        for null_connection in connections_null:
            direction = null_connection["direction"]
            degrees = direction_to_degrees_atan(direction)
            angle_percent = degrees_to_percent(degrees)
            dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_REPRESENTATION_SIZE)
            np.add(accumulated_thetas, dir_thetas, out=accumulated_thetas)

        for connection in connections:
            direction = connection["direction"]
            degrees = direction_to_degrees_atan(direction)
            angle_percent = degrees_to_percent(degrees)
            dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_REPRESENTATION_SIZE)
            np.add(accumulated_thetas, dir_thetas, out=accumulated_thetas)

            next_datapoint_name = connection["end"]
            if next_datapoint_name not in visited and next_datapoint_name is not None:
                visited.add(next_datapoint_name)
                queue.append(next_datapoint_name)

        null_thetas = []
        null_thetas = find_thetas_null_indexes(accumulated_thetas)

        for null_theta in null_thetas:
            direction_percent = get_angle_percent_from_thetas_index(null_theta, DIRECTION_REPRESENTATION_SIZE)
            direction_radians = angle_percent_to_radians(direction_percent)
            direction = generate_dxdy(direction_radians, STEP_DISTANCE)
            add_frontier(current_datapoint_name, direction)
            if return_first:
                return possible_frontiers[0]

    if return_first:
        return None

    return possible_frontiers
