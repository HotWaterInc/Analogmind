from pyparsing import empty
from src.modules.policies.testing_image_data import test_images_accuracy, process_webots_image_to_embedding, \
    squeeze_out_resnet_output

from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
import torch
import numpy as np
import math

from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.policies.utils_lib import webots_radians_to_normal
from src.utils import get_device


def check_min_distance(storage: StorageSuperset2, datapoint):
    datapoints_names = storage.get_all_datapoints()
    current_x = datapoint["params"]["x"]
    current_y = datapoint["params"]["y"]
    current_name = datapoint["name"]
    adjcent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 2)
    adjcent_names.append(current_name)

    minimum_real_distance = 1000000

    for name in datapoints_names:
        if name in adjcent_names or name == current_name:
            continue

        data = storage.get_datapoint_by_name(name)

        data_x = data["params"]["x"]
        data_y = data["params"]["y"]
        data_name = name

        real_distance = np.sqrt((current_x - data_x) ** 2 + (current_y - data_y) ** 2)
        if real_distance < minimum_real_distance:
            minimum_real_distance = real_distance

    return minimum_real_distance


def get_missing_connections_based_on_distance(storage: StorageSuperset2, datapoint, distance_threshold):
    datapoints_names = storage.get_all_datapoints()
    current_x = datapoint["params"]["x"]
    current_y = datapoint["params"]["y"]
    current_name = datapoint["name"]
    adjcent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 2)
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


def evaluate_direction_distance_validity(distance, direction, distance_sensors):
    # works only for north, needs adaptation for full rotation
    direction_percentage = direction / (2 * math.pi)
    sensors_count = len(distance_sensors)
    sensor_index_left = int(direction_percentage * sensors_count)
    sensor_index_right = (sensor_index_left + 1) % sensors_count
    wideness = 2

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


# representation details, could be anything
DIRECTION_THETAS_SIZE = 36
DISTANCE_THETAS_SIZE = 100
# max distance that can go int SDDS network
MAX_DISTANCE = 3
# what is considered the standard step. The robot distance sensors will detect null when they are smaller than step*2
STEP_DISTANCE = 0.25

# number of sampled rotations for each datapoint ( at roughly equal intervals )
ROTATIONS = 24
# how many rotations are taken into account for each datapoint ( so how many cameras shot at once )
ROTATIONS_PER_FULL = 1
# how many offsets are included in the training of the abstraction block
OFFSETS_PER_DATAPOINT = 24

# These are optimal parameters when the networks are continously trained
# Need to be found empirically
THRESHOLD_RECONSTRUCTION_ABSTRACTION_NETWORK = 0.020
THRESHOLD_NEIGHBORHOOD_NETWORK = 0.020
