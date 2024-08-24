from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
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
    wideness = 3

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


def storage_to_manifold(storage: StorageSuperset2, autoencoder: BaseAutoencoderModel):
    autoencoder.eval()
    autoencoder = autoencoder.to(get_device())

    storage.set_permutor(autoencoder)
    storage.build_permuted_data_raw_abstraction_autoencoder_manifold()
