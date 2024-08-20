import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.SDirDistState_network import run_SDirDistState, SDirDistState
from src.ai.variants.exploration.SSDir_network import run_SSDirection, SSDirNetwork, storage_to_manifold
from src.ai.variants.exploration.abstraction_block import run_abstraction_block_exploration, \
    AbstractionBlockImage
from src.ai.variants.exploration.autonomous.adjacency_detector import AdjacencyDetector, run_adjacency_network
from src.ai.variants.exploration.autonomous.exploration_autonomous_policy import abstraction_block
from src.ai.variants.exploration.autonomous.neighborhood_full import NeighborhoodNetworkThetasFull, \
    run_neighborhood_network_thetas_full
from src.ai.variants.exploration.evaluation_misc import run_tests_SSDir, run_tests_SSDir_unseen, \
    run_tests_SDirDistState
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric
from src.ai.variants.exploration.exploration_heuristics import build_find_adjacency_heursitic_raw_data, \
    find_adjacency_heuristic_neighborhood_network_thetas, build_find_adjacency_heursitic_neighborhood_network_thetas
from src.ai.variants.exploration.mutations import build_missing_connections_with_cheating
from src.ai.variants.exploration.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.ai.variants.exploration.neighborhood_network_thetas import NeighborhoodNetworkThetas, \
    run_neighborhood_network_thetas, generate_new_ai_neighborhood_thetas, DISTANCE_THETAS_SIZE, MAX_DISTANCE
from src.ai.variants.exploration.seen_network import SeenNetwork, run_seen_network
from src.ai.variants.exploration.simplified_abstract_block import run_abstraction_block_exploration_simplified, \
    AbstractionBlockSimplified
from src.ai.variants.exploration.utils import get_collected_data_image, get_collected_data_distances, \
    evaluate_direction_distance_validity, STEP_DISTANCE
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
from src.ai.variants.exploration.evaluation_exploration import print_distances_embeddings_inputs, \
    eval_distances_threshold_averages_seen_network
from src.ai.variants.exploration.autoencoder_network import run_autoencoder_network, AutoencoderExploration


def initial_setup():
    global storage, seen_network, neighborhood_distance_network, autoencoder, SSDir_network, SDirDistState_network

    storage = StorageSuperset2()
    seen_network = SeenNetwork().to(get_device())
    neighborhood_network = NeighborhoodDistanceNetwork().to(get_device())
    autoencoder = AutoencoderExploration().to(get_device())
    SSDir_network = SSDirNetwork().to(get_device())
    SDirDistState_network = SDirDistState().to(get_device())


def collect_data_generator():
    detach_robot_sample_image_inference()
    yield


def collect_distance_data_generator_with_sleep():
    time.sleep(0.05)
    detach_robot_sample_distance()
    yield


def collect_image_data_generator_with_sleep():
    time.sleep(0.1)
    detach_robot_sample_image_inference()
    yield


def create_datapoint_multiple_rotations(name: str, data: list[list[any]], coords: list[float]) -> dict[str, any]:
    datapoint = {
        "name": name,
        "data": data,
        "params": {
            "x": coords[0],
            "y": coords[1],
        }
    }
    return datapoint


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


def collect_data_rotations(rotations):
    rotation_step = 2 * math.pi / rotations
    data_arr = []
    coords = None

    for k in range(rotations):
        angle = k * rotation_step
        detach_robot_rotate_absolute(angle)
        yield

        collect_data = collect_image_data_generator_with_sleep()
        yield from collect_data
        image_embedding, angle, coords = get_collected_data_image()
        data_arr.append(image_embedding.tolist())

    return data_arr, coords


def random_walk_policy_with_rotations(random_walk_datapoints, random_walk_connections, random_walk_distances):
    ROTATIONS = 24
    rotation_step = 2 * math.pi / ROTATIONS

    data_arr = []
    coords = None
    distances_arr = []

    for k in range(ROTATIONS):
        angle = k * rotation_step
        detach_robot_rotate_absolute(angle)
        yield

        collect_data = collect_image_data_generator_with_sleep()
        yield from collect_data
        image_embedding, angle, coords = get_collected_data_image()
        data_arr.append(image_embedding.tolist())

        collect_distance_generator = collect_distance_data_generator_with_sleep()
        yield from collect_distance_generator
        distances, angle, coords = get_collected_data_distances()
        distances_arr.append(distances)

    name = f"{coords[0]:.3f}_{coords[1]:.3f}"
    datapoint = create_datapoint_multiple_rotations(name, data_arr, coords)
    datapoint_distances = create_datapoint(name, distances_arr, coords)

    random_walk_datapoints.append(datapoint)
    random_walk_distances.append(datapoint_distances)
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)

    move = move_policy()
    yield from move


def random_walk_policy(random_walk_datapoints, random_walk_connections):
    collect_data = collect_image_data_generator_with_sleep()
    yield from collect_data

    image_embedding, angle, coords = get_collected_data_image()
    name = f"{coords[0]:.3f}_{coords[1]:.3f}"
    angle_percent = angle / (2 * math.pi)

    datapoint = create_datapoint(name, image_embedding.tolist(), coords)
    random_walk_datapoints.append(datapoint)
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)

    move = move_policy()
    yield from move


def build_abstracted_data_in_storage(storage: StorageSuperset2, abstraction_block: AbstractionBlockImage):
    abstraction_block.eval()
    abstraction_block = abstraction_block.to(get_device())

    storage.set_permutor(abstraction_block)
    storage.build_permuted_data_raw_abstraction_autoencoder_manifold()


def exploration_policy_augment_data() -> any:
    initial_setup()
    global storage, seen_network, neighborhood_distance_network, autoencoder, SSDir_network, SDirDistState_network

    while True:
        random_walk_datapoints = []
        random_walk_connections = []

        random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_250_24rot.json")
        random_walk_connections = read_other_data_from_file(f"datapoints_connections_randon_walks_250_24rot.json")

        storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)
        neighborhood_distance_network = load_manually_saved_ai("neigh_full_0.25.pth")

        distance_metric = build_find_adjacency_heursitic_neighborhood_network_thetas(neighborhood_distance_network)
        # distance_metric = build_find_adjacency_heursitic_raw_data(storage)
        evaluate_distance_metric(storage, distance_metric, random_walk_datapoints)

        break


def exploration_policy_train_only() -> any:
    initial_setup()
    global storage, seen_network, neighborhood_distance_network, autoencoder, SSDir_network, SDirDistState_network

    while True:
        random_walk_datapoints = []
        random_walk_connections = []
        random_walk_distances = []

        random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_250_24rot.json")
        random_walk_connections = read_other_data_from_file(f"datapoints_connections_randon_walks_250_24rot.json")
        storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

        adjacency_network = AdjacencyDetector().to(get_device())
        adjacency_network = run_adjacency_network(adjacency_network, storage)
        save_ai_manually("adjacency_network", adjacency_network)
        # neighborhood_distance_network = NeighborhoodNetworkThetasFull().to(get_device())
        # run_neighborhood_network_thetas_full(neighborhood_distance_network, storage)
        # save_ai_manually("neigh_network", neighborhood_distance_network)
        break


def exploration_policy() -> Generator[None, None, None]:
    initial_setup()
    global storage, seen_network, neighborhood_distance_network, autoencoder, SSDir_network, SDirDistState_network

    while True:
        random_walk_datapoints = []
        random_walk_connections = []
        random_walk_distances = []

        detach_robot_teleport_absolute(0, 0)
        yield

        for step in range(0):
            generator = random_walk_policy_with_rotations(random_walk_datapoints, random_walk_connections,
                                                          random_walk_distances)
            yield from generator

        random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_250_24rot.json")
        random_walk_connections = read_other_data_from_file(f"datapoints_connections_randon_walks_250_24rot.json")

        # write_other_data_to_file(f"datapoints_connections_randon_walks_250_24rot.json",
        #                          random_walk_connections)
        # write_other_data_to_file(f"datapoints_random_walks_250_24rot.json",
        #                          random_walk_datapoints)
        # write_other_data_to_file(f"datapoints_distances_random_walks_250_24rot.json",
        #                          random_walk_distances)

        storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)
        abstraction_network = AbstractionBlockImage().to(get_device())
        run_abstraction_block_exploration(abstraction_network, storage)
        save_ai_manually("abstraction_network_simplified", abstraction_network)

        # neighborhood_network = generate_new_ai_neighborhood_thetas()
        # neighborhood_network = run_neighborhood_network_thetas(neighborhood_network, storage)
        # neighborhood_network = load_manually_saved_ai("neighborhood_network_thetas_cache.pth")
        # save_ai_manually("neighborhood_network_thetas_cache", neighborhood_network)

        # neighborhood_network = neighborhood_network.to(get_device())
        # neighborhood_network.eval()

        # find_adjacent_policy = build_find_adjacency_heursitic_raw_data(storage, seen_network)
        # find_adjacent_policy = build_find_adjacency_heursitic_neighborhood_network(neighborhood_network)
        # find_adjacent_policy = build_find_adjacency_heursitic_neighborhood_network_thetas(neighborhood_network)

        evaluate_distance_metric(storage, find_adjacent_policy, random_walk_datapoints,
                                 distance_threshold=0.35)

        # build_missing_connections_with_cheating(storage, random_walk_datapoints, distance_threshold=0.35)

        # serialize_object_other("storage_superset2_with_augmented_connections", storage)
        # print(storage.get_all_datapoints())
        # autoencoder = run_autoencoder_network(autoencoder, storage)
        # save_ai_manually("autoencoder_exploration", autoencoder)

        # autoencoder = load_manually_saved_ai("autoencoder_exploration_saved.pth")

        # storage_to_manifold(storage, autoencoder)

        # SSDir_network = load_manually_saved_ai("SSDir_network_saved.pth")
        # run_tests_SSDir(SSDir_network, storage)
        # run_tests_SSDir_unseen(SSDir_network, storage)

        # SDirDistState_network = run_SDirDistState(SDirDistState_network, storage)
        # save_ai_manually("SDirDistState_network", SDirDistState_network)
        # run_tests_SDirDistState(SDirDistState_network, storage)

        break

    yield


storage: StorageSuperset2 = None
seen_network: SeenNetwork = None
neighborhood_distance_network: NeighborhoodDistanceNetwork = None
autoencoder: AutoencoderExploration = None
SSDir_network: SSDirNetwork = None
SDirDistState_network: SDirDistState = None
