from typing import Generator

from src.ai.variants.exploration.inferences import fill_augmented_connections_distances
from src.ai.variants.exploration.metric_builders import build_find_adjacency_heursitic_raw_data, \
    build_find_adjacency_heursitic_adjacency_network
from src.ai.variants.exploration.networks.SDirDistState_network import SDirDistState
from src.ai.variants.exploration.networks.SSDir_network import SSDirNetwork
from src.ai.variants.exploration.networks.adjacency_detector import AdjacencyDetector, \
    train_adjacency_network_until_threshold
from src.ai.variants.exploration.networks.images_raw_distance_predictor import ImagesRawDistancePredictor, \
    train_images_raw_distance_predictor_until_threshold
from src.ai.variants.exploration.others.images_distance_predictor import train_images_distance_predictor_until_threshold
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork
from src.ai.variants.exploration.params import STEP_DISTANCE, ROTATIONS
from src.ai.variants.exploration.utils import get_collected_data_image, get_collected_data_distances, \
    check_direction_distance_validity_north
from src.ai.variants.exploration.utils_pure_functions import get_direction_between_datapoints
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


def initial_setup():
    global storage_raw, autoencoder, SSDir_network, SDirDistState_network, image_distance_network

    storage_raw = StorageSuperset2()
    image_distance_network = ImagesRawDistancePredictor().to(get_device())
    autoencoder = ManifoldNetwork().to(get_device())
    SSDir_network = SSDirNetwork().to(get_device())
    SDirDistState_network = SDirDistState().to(get_device())


def collect_data_generator():
    detach_robot_sample_image_inference()
    yield


def collect_distance_data_generator_without_sleep():
    detach_robot_sample_distance()
    yield


def collect_distance_data_generator_with_sleep():
    time.sleep(0.05)
    detach_robot_sample_distance()
    yield


def collect_image_data_generator_without_sleep():
    detach_robot_sample_image_inference()
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

movement_distances = np.random.uniform(0.1, 1, 500)
index = 0


def get_random_movement():
    # distance = np.random.uniform(0.1, 2)
    distance = movement_distances[index]
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
        valid = check_direction_distance_validity_north(distance, direction, distance_sensors)
        if valid:
            global index
            index += 1

    dx, dy = generate_dxdy(direction, distance)
    detach_robot_teleport_relative(dx, dy)
    yield


def add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections):
    last_datapoint = random_walk_datapoints[-1]
    added_conn = 0

    if len(random_walk_datapoints) >= 2:
        added_conn += 1
        prev_datapoint = random_walk_datapoints[-2]
        start_name = prev_datapoint["name"]
        end_name = last_datapoint["name"]
        distance = get_real_distance_between_datapoints(prev_datapoint, last_datapoint)
        direction = get_direction_between_datapoints(prev_datapoint, last_datapoint)

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
        distance = get_real_distance_between_datapoints(prev_datapoint, last_datapoint)
        direction = get_direction_between_datapoints(prev_datapoint, last_datapoint)

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
    valid = check_direction_distance_validity_north(distance, direction, distance_sensors)
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

        collect_data = collect_image_data_generator_without_sleep()
        yield from collect_data
        image_embedding, angle, coords = get_collected_data_image()
        data_arr.append(image_embedding.tolist())

        collect_distance_generator = collect_distance_data_generator_without_sleep()
        yield from collect_distance_generator
        distances, angle, coords = get_collected_data_distances()

        # sensors oriented in the direction of the robot, so we can only check the north assuming it is rotated in the direction we want
        distance = STEP_DISTANCE * 2
        direction = 0
        valid = check_direction_distance_validity_north(distance, direction, distances)
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
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)
    random_walk_connections.extend(non_valid_connections)


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


def phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps):
    for step in range(max_steps):
        collect_data = collect_current_data_and_add_connections(random_walk_datapoints, random_walk_connections)
        yield from collect_data
        move_randomly = random_move_policy()
        yield from move_randomly

        if first_walk == False:
            position_check = check_position_is_known()
            if position_check:
                break


def copy_storage(storage_to_copy: StorageSuperset2, storage_to_copy_into: StorageSuperset2):
    datapoints = storage_to_copy.get_all_datapoints()
    connections = storage_to_copy.get_all_connections_data()
    storage_to_copy_into.incorporate_new_data(datapoints, connections)
    return storage_to_copy_into


def augment_data_raw_heuristic(storage: StorageSuperset2, random_walk_datapoints) -> any:
    distance_metric = build_find_adjacency_heursitic_raw_data(storage)
    new_connections = evaluate_distance_metric(storage, distance_metric, random_walk_datapoints)
    return new_connections


def augment_data_network_heuristic(storage: StorageSuperset2, random_walk_datapoints, adjacency_network) -> any:
    adjacency_network = adjacency_network.to(get_device())
    adjacency_network.eval()

    distance_metric = build_find_adjacency_heursitic_adjacency_network(adjacency_network)
    new_connections = evaluate_distance_metric(storage, distance_metric, storage.get_all_datapoints())
    return new_connections


def exploration_policy_autonomous() -> Generator[None, None, None]:
    initial_setup()
    global storage_raw, adjacency_network, image_distance_network
    global first_walk

    detach_robot_teleport_absolute(0, 0)
    yield

    exploring = True
    first_walk = True
    iter = 0

    while exploring:
        iter += 1
        random_walk_datapoints = []
        random_walk_connections = []

        print(f"EXPLORING RANDOM WALK ITER {iter}")
        if first_walk:
            print("FIRST TIME")

        yield from phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps=10)
        first_walk = False
        flag_data_authenticity(random_walk_connections)
        storage_raw.incorporate_new_data(random_walk_datapoints, random_walk_connections)
        print("FINISHED RANDOM WALK")

        write_other_data_to_file(f"datapoints_random_walks_iter{iter}.json", random_walk_datapoints)
        write_other_data_to_file(f"datapoints_connections_random_walks_iter{iter}.json", random_walk_connections)

        print("TRAINING ADJACENCY NETWORK")
        adjacency_network = train_adjacency_network_until_threshold(adjacency_network, storage_raw)
        print("AUGMENTING DATA")
        new_connections_raw = augment_data_raw_heuristic(storage_raw, random_walk_datapoints)
        new_connections_adjacency_network = augment_data_network_heuristic(storage_raw, random_walk_datapoints,
                                                                           adjacency_network)

        total_connections_found = []
        total_connections_found.extend(new_connections_raw)
        total_connections_found.extend(new_connections_adjacency_network)
        flag_data_authenticity(total_connections_found)

        write_other_data_to_file(f"additional_found_connections_rawh_random_walks_iter{iter}.json", new_connections_raw)
        write_other_data_to_file(f"additional_found_connections_networkh_random_walks_iter{iter}.json",
                                 new_connections_adjacency_network)

        evaluate_distance_metric_on_already_found_connections(storage_raw, random_walk_datapoints,
                                                              total_connections_found)
        print("FINISHING AUGMENTING CONNECTIONS")
        print("TRAINING DISTANCES NETWORK")
        image_distance_network = train_images_raw_distance_predictor_until_threshold(image_distance_network,
                                                                                     storage_raw)
        print("ADDING SYNTHETIC DISTANCES")
        total_new_connections_filled = fill_augmented_connections_distances(
            additional_connections=total_connections_found,
            storage=storage_raw,
            image_distance_network=image_distance_network)

        storage_raw.incorporate_new_data([], total_new_connections_filled)
        write_other_data_to_file(f"additional_found_total_connections_distance_augmented_iter{iter}.json",
                                 total_new_connections_filled)
        print("SKIPPING DATA PURGING")

        break

    yield


storage_raw: StorageSuperset2 = None
storage_manifold: StorageSuperset2 = None

adjacency_network: AdjacencyDetector = None
image_distance_network: ImagesRawDistancePredictor = None

autoencoder: ManifoldNetwork = None
direction_network_SSD: nn.Module = None
direction_network_SDirDistS: nn.Module = None

global_register1 = None
global_register2 = None

first_walk = True
