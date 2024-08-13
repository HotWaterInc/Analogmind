import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration_normal.SDirDistState_network import run_SDirDistState, SDirDistState
from src.ai.variants.exploration_normal.SSDir_network import run_SSDirection, SSDirNetwork, storage_to_manifold
from src.ai.variants.exploration_normal.evaluation_misc import run_tests_SSDir, run_tests_SSDir_unseen, \
    run_tests_SDirDistState
from src.ai.variants.exploration_normal.mutations import build_missing_connections_with_cheating
from src.ai.variants.exploration_normal.neighborhood_network import NeighborhoodNetwork, run_neighborhood_network
from src.ai.variants.exploration_normal.neighborhood_network_thetas import NeighborhoodNetworkThetas, \
    run_neighborhood_network_thetas, generate_new_ai_neighborhood_thetas, DISTANCE_THETAS_SIZE, MAX_DISTANCE
from src.ai.variants.exploration_normal.seen_network import SeenNetwork, run_seen_network
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.save_load_handlers.data_handle import write_other_data_to_file, serialize_object_other, \
    deserialize_object_other
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_sample_image, \
    detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute, \
    detach_robot_rotate_continuous_absolute, detach_robot_forward_continuous, detach_robot_sample_image_inference
import threading
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai, load_custom_ai, load_other_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import *
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super
from src.modules.policies.data_collection import get_position, get_angle
from src.modules.policies.testing_image_data import test_images_accuracy, process_webots_image_to_embedding, \
    squeeze_out_resnet_output
from src.modules.policies.utils_lib import webots_radians_to_normal, radians_to_degrees
import torch
from src.utils import get_device
from src.ai.variants.exploration_normal.evaluation_exploration import print_distances_embeddings_inputs, \
    eval_distances_threshold_averages, evaluate_distance_metric

from src.ai.variants.exploration_normal.autoencoder_network import run_autoencoder_network, AutoencoderExploration


def initial_setup():
    global storage, seen_network, neighborhood_network, autoencoder, SSDir_network, SDirDistState_network

    storage = StorageSuperset2()
    seen_network = SeenNetwork().to(get_device())
    neighborhood_network = NeighborhoodNetwork().to(get_device())
    autoencoder = AutoencoderExploration().to(get_device())
    SSDir_network = SSDirNetwork().to(get_device())
    SDirDistState_network = SDirDistState().to(get_device())


def collect_data_generator():
    detach_robot_sample_image_inference()
    yield


def collect_data_generator_with_sleep():
    time.sleep(0.1)
    detach_robot_sample_image_inference()
    yield


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


def create_datapoint(name: str, data: List[any], coords: List[float]) -> Dict[str, any]:
    datapoint = {
        "name": name,
        "data": [data],
        "params": {
            "x": coords[0],
            "y": coords[1],
        }
    }
    return datapoint


movement_type = 0


def get_random_movement():
    distance = np.random.uniform(0.05, 1)
    direction = np.random.uniform(0, 2 * math.pi)

    return distance, direction


def generate_dxdy(direction, distance):
    dx = -distance * math.sin(direction)
    dy = distance * math.cos(direction)

    return dx, dy


def distance_sensors_transform(distance):
    # formula is roughly sensor_distance = 10 * distance + 2.5
    return (distance - 2.5) / 10


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


def move_policy():
    # collects sensor data and finds a valid move direction
    detach_robot_sample_distance()
    yield
    distance_sensors, angle, coords = get_collected_data_distances()

    valid = False
    distance, direction = 0, 0
    while not valid:
        distance, direction = get_random_movement()
        valid = evaluate_direction_distance_validity(distance, direction, distance_sensors)

    dx, dy = generate_dxdy(direction, distance)
    # print("Direction", direction / (2 * math.pi), "dx dy", dx, dy)
    detach_robot_teleport_relative(dx, dy)
    yield


def _get_distance(coords1: any, coords2: any) -> float:
    x1, y1 = coords1
    x2, y2 = coords2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _get_distance_datapoints(datapoint1: Dict[str, any], datapoint2: Dict[str, any]) -> float:
    coords1 = datapoint1["params"]["x"], datapoint1["params"]["y"]
    coords2 = datapoint2["params"]["x"], datapoint2["params"]["y"]
    return _get_distance(coords1, coords2)


def _get_direction_datapoints(datapoint1: Dict[str, any], datapoint2: Dict[str, any]) -> tuple[float, float]:
    coords1 = datapoint1["params"]["x"], datapoint1["params"]["y"]
    coords2 = datapoint2["params"]["x"], datapoint2["params"]["y"]
    direction_vector = (coords2[0] - coords1[0], coords2[1] - coords1[1])
    return direction_vector


def add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections):
    last_datapoint = random_walk_datapoints[-1]
    added_conn = 0

    if len(random_walk_datapoints) >= 2:
        added_conn += 1
        prev_datapoint = random_walk_datapoints[-2]
        start_name = prev_datapoint["name"]
        end_name = last_datapoint["name"]
        distance = _get_distance_datapoints(prev_datapoint, last_datapoint)
        direction = _get_direction_datapoints(prev_datapoint, last_datapoint)

        connection = {
            "start": start_name,
            "end": end_name,
            "distance": distance,
            "direction": direction
        }
        random_walk_connections.append(connection)

    if len(random_walk_datapoints) >= 3:
        added_conn += 1
        prev_datapoint = random_walk_datapoints[-3]
        start_name = prev_datapoint["name"]
        end_name = last_datapoint["name"]
        distance = _get_distance_datapoints(prev_datapoint, last_datapoint)
        direction = _get_direction_datapoints(prev_datapoint, last_datapoint)

        connection = {
            "start": start_name,
            "end": end_name,
            "distance": distance,
            "direction": direction
        }
        random_walk_connections.append(connection)


def relative_difference(a, b):
    return abs(a - b) / ((a + b) / 2)


def build_find_adjacency_heursitic_neighborhood_network(
        neighborhood_network: NeighborhoodNetwork):
    def find_adjacency_heursitic_augmented(storage: StorageSuperset2, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_neighborhood_network(storage, neighborhood_network, datapoint)

    return find_adjacency_heursitic_augmented


def build_find_adjacency_heursitic_neighborhood_network_thetas(
        neighborhood_network: NeighborhoodNetworkThetas):
    def find_adjacency_heursitic_augmented(storage: StorageSuperset2, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_neighborhood_network_thetas(storage, neighborhood_network, datapoint)

    return find_adjacency_heursitic_augmented


def find_adjacency_heuristic_neighborhood_network_thetas(storage: StorageSuperset2,
                                                         neighborhood_network: NeighborhoodNetworkThetas,
                                                         datapoint: Dict[str, any]):
    current_name = datapoint["name"]
    current_data = datapoint["data"]
    current_data = torch.tensor(current_data).to(get_device())

    datapoints_names = storage.get_all_datapoints()
    neighborhood_network.eval()
    neighborhood_network = neighborhood_network.to(get_device())

    found = 0
    for name in datapoints_names:
        if name == current_name:
            continue

        existing_data = storage.get_datapoint_data_tensor_by_name(name).to(get_device())

        distance_thetas = neighborhood_network(current_data,
                                               existing_data).squeeze()
        distance_percent = distance_thetas_to_distance_percent(distance_thetas)
        distance_percent *= MAX_DISTANCE

        if distance_percent < 0.35:
            found = 1

    return found


def find_adjacency_heuristic_neighborhood_network(storage: StorageSuperset2, neighborhood_network: NeighborhoodNetwork,
                                                  datapoint: Dict[str, any]):
    current_name = datapoint["name"]
    current_data = datapoint["data"]
    current_data = torch.tensor(current_data).to(get_device())

    datapoints_names = storage.get_all_datapoints()
    neighborhood_network.eval()
    neighborhood_network = neighborhood_network.to(get_device())

    found = 0
    for name in datapoints_names:
        if name == current_name:
            continue
        existing_data = storage.get_datapoint_data_tensor_by_name(name).to(get_device())

        expected_distance = neighborhood_network(current_data,
                                                 existing_data).squeeze()

        if expected_distance < 0.5:
            found = 1

    return found


def build_find_adjacency_heursitic_raw_data(storage: StorageSuperset2, seen_network: SeenNetwork):
    distance_embeddings, distance_data = eval_distances_threshold_averages(storage, seen_network,
                                                                           real_distance_threshold=0.35)
    # distance_embeddings *= 1
    distance_data *= 1

    def find_adjacency_heursitic_augmented(storage: StorageSuperset2, datapoint: Dict[str, any]):
        return find_adjacency_heuristic_raw_data(storage, seen_network, datapoint, distance_embeddings, distance_data)

    return find_adjacency_heursitic_augmented


def find_adjacency_heuristic_raw_data(storage: StorageSuperset2, seen_network: SeenNetwork, datapoint: Dict[str, any],
                                      distance_embeddings, distance_data):
    current_name = datapoint["name"]
    current_data = datapoint["data"][0]
    current_data = torch.tensor(current_data).to(get_device())
    datapoints_names = storage.get_all_datapoints()
    adjcent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 2)
    adjcent_names.append(current_name)

    found = 0

    current_data_arr = []
    other_datapoints_data_arr = []
    for name in datapoints_names:
        if name in adjcent_names or name == current_name:
            continue

        existing_data = storage.get_datapoint_data_tensor_by_name_cached(name).squeeze()
        other_datapoints_data_arr.append(existing_data)
        current_data_arr.append(current_data)

    current_data_arr = torch.stack(current_data_arr).to(get_device())
    other_datapoints_data_arr = torch.stack(other_datapoints_data_arr).to(get_device())
    norm_distance = torch.norm(current_data_arr - other_datapoints_data_arr, p=2, dim=1)
    length = len(norm_distance)

    for i in range(length):
        distance_data_i = norm_distance[i]
        if distance_data_i < distance_data:
            found = 1

    return found


def display_sensor_data():
    detach_robot_sample_distance()
    yield
    distance_sensors, angle, coords = get_collected_data_distances()

    valid = False
    distance, direction = get_random_movement()
    distance = 0.5
    direction = 3.14
    valid = evaluate_direction_distance_validity(distance, direction, distance_sensors)
    print(valid)


def random_walk_policy(random_walk_datapoints, random_walk_connections):
    collect_data = collect_data_generator_with_sleep()
    yield from collect_data

    image_embedding, angle, coords = get_collected_data_image()
    name = f"{coords[0]:.3f}_{coords[1]:.3f}"
    angle_percent = angle / (2 * math.pi)
    datapoint = create_datapoint(name, image_embedding.tolist(), coords)
    random_walk_datapoints.append(datapoint)
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)

    move = move_policy()
    yield from move


def exploration_policy() -> Generator[None, None, None]:
    initial_setup()
    global storage, seen_network, neighborhood_network, autoencoder, SSDir_network, SDirDistState_network

    while True:
        random_walk_datapoints = []
        random_walk_connections = []

        detach_robot_teleport_absolute(0, 0)
        yield

        for step in range(0):
            generator = random_walk_policy(random_walk_datapoints, random_walk_connections)
            yield from generator

        random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_500.json")
        random_walk_connections = read_other_data_from_file(f"datapoints_connections_randon_walks_500.json")
        # write_other_data_to_file(f"datapoints_connections_randon_walks_500.json",
        #                          random_walk_connections)
        # write_other_data_to_file(f"datapoints_random_walks_500.json",
        #                          random_walk_datapoints)

        storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

        # neighborhood_network = generate_new_ai_neighborhood_thetas()
        # neighborhood_network = run_neighborhood_network_thetas(neighborhood_network, storage)
        # neighborhood_network = load_manually_saved_ai("neighborhood_network_thetas_cache.pth")
        # save_ai_manually("neighborhood_network_thetas_cache", neighborhood_network)

        # neighborhood_network = neighborhood_network.to(get_device())
        # neighborhood_network.eval()

        # find_adjacent_policy = build_find_adjacency_heursitic_raw_data(storage, seen_network)
        # find_adjacent_policy = build_find_adjacency_heursitic_neighborhood_network(neighborhood_network)
        # find_adjacent_policy = build_find_adjacency_heursitic_neighborhood_network_thetas(neighborhood_network)
        # evaluate_distance_metric(storage, find_adjacent_policy, random_walk_datapoints,
        #                          distance_threshold=0.35)

        build_missing_connections_with_cheating(storage, random_walk_datapoints, distance_threshold=0.35)

        # serialize_object_other("storage_superset2_with_augmented_connections", storage)
        # print(storage.get_all_datapoints())
        # autoencoder = run_autoencoder_network(autoencoder, storage)
        # save_ai_manually("autoencoder_exploration", autoencoder)

        autoencoder = load_manually_saved_ai("autoencoder_exploration_saved.pth")

        storage_to_manifold(storage, autoencoder)

        # SSDir_network = load_manually_saved_ai("SSDir_network_saved.pth")
        # run_tests_SSDir(SSDir_network, storage)
        # run_tests_SSDir_unseen(SSDir_network, storage)

        SDirDistState_network = run_SDirDistState(SDirDistState_network, storage)
        save_ai_manually("SDirDistState_network", SDirDistState_network)
        run_tests_SDirDistState(SDirDistState_network, storage)

        break

    yield


storage: StorageSuperset2 = None
seen_network: SeenNetwork = None
neighborhood_network: NeighborhoodNetwork = None
autoencoder: AutoencoderExploration = None
SSDir_network: SSDirNetwork = None
SDirDistState_network: SDirDistState = None
