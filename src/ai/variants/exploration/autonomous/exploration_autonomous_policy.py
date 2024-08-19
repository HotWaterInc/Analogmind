import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.SDirDistState_network import run_SDirDistState, SDirDistState
from src.ai.variants.exploration.SSDir_network import run_SSDirection, SSDirNetwork, storage_to_manifold
from src.ai.variants.exploration.abstraction_block import run_abstraction_block_exploration, \
    AbstractionBlockImage, run_abstraction_block_exploration_until_threshold
from src.ai.variants.exploration.evaluation_misc import run_tests_SSDir, run_tests_SSDir_unseen, \
    run_tests_SDirDistState
from src.ai.variants.exploration.exploration_heuristics import build_find_adjacency_heursitic_raw_data, \
    build_augmented_connections, fill_augmented_connections_distances, filter_surplus_datapoints
from src.ai.variants.exploration.mutations import build_missing_connections_with_cheating
from src.ai.variants.exploration.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network, run_neighborhood_network_until_threshold
from src.ai.variants.exploration.neighborhood_network_thetas import NeighborhoodNetworkThetas, \
    run_neighborhood_network_thetas, generate_new_ai_neighborhood_thetas
from src.ai.variants.exploration.seen_network import SeenNetwork, run_seen_network
from src.ai.variants.exploration.utils import get_collected_data_image, get_collected_data_distances, \
    evaluate_direction_distance_validity, ROTATIONS, STEP_DISTANCE, THRESHOLD_RECONSTRUCTION_ABSTRACTION_NETWORK, \
    THRESHOLD_NEIGHBORHOOD_NETWORK
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
    eval_distances_threshold_averages_seen_network, eval_distances_threshold_averages_raw_data
from src.ai.variants.exploration.autoencoder_network import run_autoencoder_network, AutoencoderExploration
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric


def initial_setup():
    global storage_raw, neighborhood_distance_network, autoencoder, SSDir_network, SDirDistState_network

    storage_raw = StorageSuperset2()
    neighborhood_distance_network = NeighborhoodDistanceNetwork().to(get_device())
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


def generate_dxdy(direction: float, distance):
    # direction in radians
    dx = -distance * math.sin(direction)
    dy = distance * math.cos(direction)

    return dx, dy


def check_position_is_known():
    global abstraction_block
    # checks current position by reconstruction loss, and similar positioning heuristic
    # TODO IMPLEMENT
    return False


def random_move_policy():
    """
    Constrained only by distance sensors
    """

    global walk_directions_stack
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


def build_find_adjacency_heursitic_neighborhood_network(
        neighborhood_network: NeighborhoodDistanceNetwork):
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


def find_adjacency_heuristic_neighborhood_network(storage: StorageSuperset2,
                                                  neighborhood_network: NeighborhoodDistanceNetwork,
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


def null_connections_to_raw_connections_data(name, null_connections):
    non_valid_connections = []
    for null_connection in null_connections:
        angle = null_connection["angle"]
        valid = null_connection["valid"]
        # if we bump into something we create a null connections
        if valid == False:
            dx, dy = generate_dxdy(angle, STEP_DISTANCE * 2)
            direction = [dx, dy]
            distance = STEP_DISTANCE * 2
            connection = {
                "start": name,
                "end": None,
                "distance": distance,
                "direction": direction
            }
            non_valid_connections.append(connection)

    return non_valid_connections


def collect_data_rotations_and_create_datapoint():
    rotation_step = 2 * math.pi / ROTATIONS
    data_arr = []
    coords = None
    null_connections_tests = []
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

        # sensors oriented in the direction of the robot, so we can only check the north assuming it is rotated in the direction we want
        distance = STEP_DISTANCE * 2
        direction = 0
        valid = evaluate_direction_distance_validity(distance, direction, distances)
        print("At rotation", angle, " the valid param is", valid)
        null_connections_tests.append({
            "angle": angle,
            "valid": valid
        })

    name = f"{coords[0]:.3f}_{coords[1]:.3f}"
    datapoint = create_datapoint_multiple_rotations(name, data_arr, coords)

    global global_register1, global_register2
    global_register1 = datapoint
    global_register2 = null_connections_tests


def return_global_buffer2() -> any:
    global global_register2
    aux = global_register2
    global_register2 = None
    return aux


def return_global_buffer1() -> any:
    global global_register1
    aux = global_register1
    global_register1 = None
    return aux


def collect_current_data_and_add_connections(random_walk_datapoints, random_walk_connections):
    global global_register1
    yield from collect_data_rotations_and_create_datapoint()

    datapoint = return_global_buffer1()
    null_connections_test = return_global_buffer2()

    random_walk_datapoints.append(datapoint)
    non_valid_connections = null_connections_to_raw_connections_data(datapoint["name"], null_connections_test)
    print("Appended non valid connections numbergin", len(non_valid_connections))
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)
    random_walk_connections.extend(non_valid_connections)


def walk_random_unconstrained_policy_with_rotations():
    pass


def random_walk_policy(random_walk_datapoints, random_walk_connections):
    collect_data = collect_image_data_generator_with_sleep()
    yield from collect_data

    image_embedding, angle, coords = get_collected_data_image()
    name = f"{coords[0]:.3f}_{coords[1]:.3f}"
    angle_percent = angle / (2 * math.pi)

    datapoint = create_datapoint(name, image_embedding.tolist(), coords)
    random_walk_datapoints.append(datapoint)
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)

    move = random_move_policy()
    yield from move


def phase_explore(random_walk_datapoints, random_walk_connections, first_walk=False):
    for step in range(20):
        collect_data = collect_current_data_and_add_connections(random_walk_datapoints, random_walk_connections)
        yield from collect_data
        move_randomly = random_move_policy()
        yield from move_randomly

        if first_walk == False:
            position_check = check_position_is_known()
            if position_check:
                break


def train_abstraction_block():
    """
    Train abstraction block, the most important part. It gives the basis on which all other networks build upon
    That means the filtered position from the data
    """
    global abstraction_block, storage_raw
    abstraction_block = run_abstraction_block_exploration_until_threshold(abstraction_block, storage_raw,
                                                                          THRESHOLD_RECONSTRUCTION_ABSTRACTION_NETWORK)


def train_neighborhood_network():
    global neighborhood_distance_network, storage_abstracted
    neighborhood_distance_network = run_neighborhood_network_until_threshold(neighborhood_distance_network,
                                                                             storage_abstracted,
                                                                             THRESHOLD_NEIGHBORHOOD_NETWORK)


def train_manifold_network():
    global storage_manifold, autoencoder, storage_abstracted
    autoencoder = run_autoencoder_network(autoencoder, storage_manifold)


def train_navigation_networks():
    global SSDir_network, SDirDistState_network, storage_abstracted

    SSDir_network = run_SSDirection(SSDir_network, storage_abstracted)
    SDirDistState_network = run_SDirDistState(SDirDistState_network, storage_abstracted)


def build_abstracted_data_in_storage(storage: StorageSuperset2, abstraction_block: AbstractionBlockImage):
    abstraction_block.eval()
    abstraction_block = abstraction_block.to(get_device())

    storage.set_permutor(autoencoder)
    storage.build_permuted_data_raw_abstraction_autoencoder_manifold()


def copy_storage(storage_to_copy: StorageSuperset2, storage_to_copy_into: StorageSuperset2):
    datapoints = storage_to_copy.get_all_datapoints()
    connections = storage_to_copy.get_all_connections_data()
    storage_to_copy_into.incorporate_new_data(datapoints, connections)
    return storage_to_copy_into


def exploration_policy() -> Generator[None, None, None]:
    initial_setup()
    global storage_raw, storage_abstracted, abstraction_block, neighborhood_distance_network, autoencoder, SSDir_network, SDirDistState_network

    detach_robot_teleport_absolute(0, 0)
    yield

    exploring = True
    first_walk = True

    while exploring:
        random_walk_datapoints = []
        random_walk_connections = []

        yield from phase_explore(random_walk_datapoints, random_walk_connections, first_walk)
        first_walk = False
        storage_raw.incorporate_new_data(random_walk_datapoints, random_walk_connections)
        # trains basis of all other networks
        train_abstraction_block()

        # builds abstracted data in storage
        storage_abstracted = StorageSuperset2()
        storage_abstracted = copy_storage(storage_raw, storage_abstracted)
        build_abstracted_data_in_storage(storage_abstracted, abstraction_block)

        # train neighborhood network on abstracted data
        train_neighborhood_network()
        find_adjacency_heuristic = build_find_adjacency_heursitic_raw_data(storage_abstracted)
        additional_connections = build_augmented_connections(storage_abstracted, find_adjacency_heuristic,
                                                             STEP_DISTANCE * 1.5)
        augmented_connections = fill_augmented_connections_distances(additional_connections, storage_abstracted,
                                                                     neighborhood_distance_network)
        storage_abstracted.incorporate_new_data([], augmented_connections)
        filter_surplus_datapoints(storage_abstracted)
        train_manifold_network()
        # transform storage again into manifold
        train_navigation_networks()

        # find best place to explore next
        # navigate to that place
        # start all over again

        break
    yield


storage_raw: StorageSuperset2 = None
storage_abstracted: StorageSuperset2 = None
storage_manifold: StorageSuperset2 = None
abstraction_block: AbstractionBlockImage = None
neighborhood_distance_network: NeighborhoodDistanceNetwork = None

autoencoder: AutoencoderExploration = None
direction_network_SSD: nn.Module = None
direction_network_SDirDistS: nn.Module = None

global_register1 = None
global_register2 = None
