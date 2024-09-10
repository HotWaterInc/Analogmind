from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.params import DIRECTION_THETAS_SIZE, STEP_DISTANCE, STEP_DISTANCE_BASIC_STEP, \
    EXPERIMENTAL_BINARY_SIZE, MANIFOLD_SIZE
from src.ai.variants.exploration.utils_pure_functions import direction_to_degrees_atan, degrees_to_percent, \
    angle_percent_to_thetas_normalized_cached, direction_thetas_to_radians, find_thetas_null_indexes, \
    get_angle_percent_from_thetas_index, generate_dxdy, angle_percent_to_radians
from src.modules.policies.navigation8x8_v1_distance import radians_to_degrees
from src.modules.policies.testing_image_data import process_webots_image_to_embedding, \
    squeeze_out_resnet_output
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2

import torch
import numpy as np
import math

from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.policies.utils_lib import webots_radians_to_normal
from src.utils import get_device


def get_missing_connections_based_on_distance(storage: StorageSuperset2, datapoint, distance_threshold):
    datapoints_names = storage.get_all_datapoints()
    current_x = datapoint["params"]["x"]
    current_y = datapoint["params"]["y"]
    current_name = datapoint["name"]
    adjcent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(current_name, 2)
    adjcent_names.append(current_name)

    found_connections = []

    for name in datapoints_names:
        if name in adjcent_names or name == current_name:
            continue

        data = storage.get_datapoint_by_name(name)

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

    print(rotation_percentage)
    print(new_distance_sensors)
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


def check_direction_distance_validity_north(distance, direction, distance_sensors):
    # works only for north, needs adaptation for full rotation
    direction_percentage = direction / (2 * math.pi)
    sensors_count = len(distance_sensors)
    sensor_index_left = int(direction_percentage * sensors_count)
    sensor_index_right = (sensor_index_left + 1) % sensors_count
    wideness = 4

    for offset in range(wideness):
        left_index = sensor_index_left - offset
        right_index = sensor_index_right + offset

        if left_index < 0:
            left_index = sensors_count + left_index

        if right_index >= sensors_count:
            right_index = right_index - sensors_count

        sensor_left_distance = distance_sensors_transform(distance_sensors[left_index])
        sensor_right_distance = distance_sensors_transform(distance_sensors[right_index])

        if sensor_left_distance < distance or sensor_right_distance < distance:
            return False

    return True


def distance_sensors_transform(distance):
    # formula is roughly sensor_distance = 10 * distance + 2.5
    return (distance - 2.5) / 10


def get_collected_data_distances() -> tuple[torch.Tensor, float, list[float]]:
    global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
    buffer = global_data_buffer.buffer
    distances = buffer["data"]
    empty_global_data_buffer()

    nd_array_data = np.array(distances)
    angle = buffer["params"]["angle"]
    x = buffer["params"]["x"]
    y = buffer["params"]["y"]
    coords = [
        round(x, 3),
        round(y, 3)
    ]
    angle = webots_radians_to_normal(angle)

    return distances, angle, coords


def get_collected_data_image() -> tuple[torch.Tensor, float, list[float]]:
    global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
    buffer = global_data_buffer.buffer
    image_data = buffer["data"]
    empty_global_data_buffer()

    nd_array_data = np.array(image_data)
    angle = buffer["params"]["angle"]
    x = buffer["params"]["x"]
    y = buffer["params"]["y"]
    coords = [
        round(x, 3),
        round(y, 3)
    ]
    # trim coords to 3rd decimal

    angle = webots_radians_to_normal(angle)

    current_embedding = process_webots_image_to_embedding(nd_array_data).to(get_device())
    current_embedding = squeeze_out_resnet_output(current_embedding)

    return current_embedding, angle, coords


def value_to_binary_array(value: int, bits_count: int) -> list:
    # Create a list to store the binary digits
    binary_array = [0] * bits_count

    # Iterate through the bits
    for i in range(bits_count):
        # Use bitwise AND to check if the bit is set
        if value & (1 << i):
            binary_array[bits_count - 1 - i] = 1
    return binary_array


def data_to_binary_data(data: torch.Tensor, index: int):
    new_data = torch.zeros(data.shape[0], EXPERIMENTAL_BINARY_SIZE)
    for idx, val in enumerate(data):
        new_data[idx] = torch.tensor(value_to_binary_array(index, EXPERIMENTAL_BINARY_SIZE))

    return new_data


def data_to_random_manifold(data, index: int):
    new_data = torch.zeros(data.shape[0], MANIFOLD_SIZE)
    random_manifold = torch.rand(MANIFOLD_SIZE)

    for idx, val in enumerate(data):
        new_data[idx] = random_manifold

    return new_data


def storage_to_binary_data(storage: StorageSuperset2):
    storage.set_transformation(data_to_binary_data)
    storage.transform_raw_data()


def storage_to_random_manifold(storage: StorageSuperset2):
    storage.set_transformation(data_to_random_manifold)
    storage.transform_raw_data()


def storage_to_manifold(storage: StorageSuperset2, manifold_network: BaseAutoencoderModel):
    manifold_network.eval()
    manifold_network = manifold_network.to(get_device())

    storage.set_transformation(manifold_network)
    storage.build_permuted_data_raw_abstraction_autoencoder_manifold()


def check_datapoint_connections_completeness(storage: StorageSuperset2, datapoint_name: str):
    direction_thetas_accumulated = np.zeros(DIRECTION_THETAS_SIZE)
    connections = storage.get_datapoint_adjacent_connections_direction_filled(datapoint_name)
    connections_non_null = [connection for connection in connections if connection["end"] is not None]
    connections = connections_non_null

    for connection in connections:
        direction = connection["direction"]
        degrees = direction_to_degrees_atan(direction)
        angle_percent = degrees_to_percent(degrees)
        dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_THETAS_SIZE)

        np.add(direction_thetas_accumulated, dir_thetas, out=direction_thetas_accumulated)

    has_zeros = False
    for theta in direction_thetas_accumulated:
        if theta == 0:
            has_zeros = True
            break

    connection_completeness = not has_zeros and len(connections) >= 5

    return connection_completeness


def check_datapoint_density(storage: StorageSuperset2, datapoint_name: str):
    connections = storage.get_datapoint_adjacent_connections_direction_filled(datapoint_name)
    connections_non_null = [connection for connection in connections if connection["end"] is not None]
    connections = connections_non_null
    connections_null = storage.get_datapoint_adjacent_connections_null_connections(datapoint_name)

    return len(connections) > 20 and len(connections_null) == 0


def find_frontier_all_datapoint_and_direction(storage: StorageSuperset2, return_first: bool = False,
                                              starting_point=None):
    """
    Finds empty connections gaps and goes there to explore, starting from the current datapoint
    If no gaps are found it means the entire map was explored
    """

    current_datapoint_name = None
    if starting_point is not None:
        current_datapoint_name = starting_point
    else:
        current_datapoint_name = storage.get_all_datapoints()[0]

    queue = [current_datapoint_name]
    visited = set()
    visited.add(current_datapoint_name)

    possible_frontiers = []

    def add_frontier(datapoint_name, direction):
        distance = STEP_DISTANCE_BASIC_STEP
        possible_frontiers.append({
            "start": datapoint_name,
            "end": None,
            "direction": direction,
            "distance": distance
        })

    while queue:
        current_datapoint_name = queue.pop(0)
        connections = storage.get_datapoint_adjacent_connections_direction_filled(current_datapoint_name)
        connections_null = storage.get_datapoint_adjacent_connections_null_connections(current_datapoint_name)

        accumulated_thetas = np.zeros(DIRECTION_THETAS_SIZE)

        for null_connection in connections_null:
            direction = null_connection["direction"]
            degrees = direction_to_degrees_atan(direction)
            angle_percent = degrees_to_percent(degrees)
            dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_THETAS_SIZE)
            np.add(accumulated_thetas, dir_thetas, out=accumulated_thetas)

        for connection in connections:
            direction = connection["direction"]
            degrees = direction_to_degrees_atan(direction)
            angle_percent = degrees_to_percent(degrees)
            dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_THETAS_SIZE)
            np.add(accumulated_thetas, dir_thetas, out=accumulated_thetas)

            next_datapoint_name = connection["end"]
            if next_datapoint_name not in visited and next_datapoint_name is not None:
                visited.add(next_datapoint_name)
                queue.append(next_datapoint_name)

        null_thetas = []
        null_thetas = find_thetas_null_indexes(accumulated_thetas)

        for null_theta in null_thetas:
            direction_percent = get_angle_percent_from_thetas_index(null_theta, DIRECTION_THETAS_SIZE)
            direction_radians = angle_percent_to_radians(direction_percent)
            direction = generate_dxdy(direction_radians, STEP_DISTANCE_BASIC_STEP)
            add_frontier(current_datapoint_name, direction)
            if return_first:
                return possible_frontiers[0]

    if return_first:
        return None

    return possible_frontiers


def find_frontier_closest_datapoint_and_direction(storage: StorageSuperset2, current_datapoint_name):
    """
    Finds empty connections gaps and goes there to explore, starting from the current datapoint
    If no gaps are found it means the entire map was explored
    """

    queue = [current_datapoint_name]
    visited = set()
    visited.add(current_datapoint_name)

    while queue:
        current_datapoint_name = queue.pop(0)
        connections = storage.get_datapoint_adjacent_connections_direction_filled(current_datapoint_name)
        connections_null = storage.get_datapoint_adjacent_connections_null_connections(current_datapoint_name)

        accumulated_thetas = np.zeros(DIRECTION_THETAS_SIZE)

        for null_connection in connections_null:
            direction = null_connection["direction"]
            degrees = direction_to_degrees_atan(direction)
            angle_percent = degrees_to_percent(degrees)
            dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_THETAS_SIZE)
            np.add(accumulated_thetas, dir_thetas, out=accumulated_thetas)

        for connection in connections:
            direction = connection["direction"]
            degrees = direction_to_degrees_atan(direction)
            angle_percent = degrees_to_percent(degrees)
            dir_thetas = angle_percent_to_thetas_normalized_cached(angle_percent, DIRECTION_THETAS_SIZE)
            np.add(accumulated_thetas, dir_thetas, out=accumulated_thetas)

            next_datapoint_name = connection["end"]
            if next_datapoint_name not in visited and next_datapoint_name is not None:
                visited.add(next_datapoint_name)
                queue.append(next_datapoint_name)

        null_thetas = find_thetas_null_indexes(accumulated_thetas)

        if len(null_thetas) > 0:
            direction_percent = get_angle_percent_from_thetas_index(null_thetas[0], DIRECTION_THETAS_SIZE)
            direction_radians = angle_percent_to_radians(direction_percent)
            direction = generate_dxdy(direction_radians, STEP_DISTANCE_BASIC_STEP)
            return current_datapoint_name, direction

    return None, None
