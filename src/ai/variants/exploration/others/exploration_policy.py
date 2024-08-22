from src.ai.variants.exploration.SDirDistState_network import SDirDistState
from src.ai.variants.exploration.SSDir_network import SSDirNetwork
from src.ai.variants.exploration.abstraction_block import AbstractionBlockImage
from src.ai.variants.exploration.abstraction_block_second_trial import run_abstraction_block_second_trial, \
    AbstractionBlockSecondTrial
from src.ai.variants.exploration.networks.adjacency_detector import AdjacencyDetector, train_adjacency_network
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric
from src.ai.variants.exploration.exploration_heuristics import build_find_adjacency_heursitic_raw_data, \
    build_find_adjacency_heursitic_adjacency_network
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork
from src.ai.variants.exploration.seen_network import SeenNetwork
from src.ai.variants.exploration.utils import get_collected_data_image, get_collected_data_distances, \
    check_direction_distance_validity
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_sample_image_inference
import time
from src.modules.save_load_handlers.ai_models_handle import save_ai_manually
from src.ai.runtime_data_storage.storage_superset2 import *
from typing import List, Dict
from src.utils import get_device
from src.ai.variants.exploration.autoencoder_network import AutoencoderExploration


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
        valid = check_direction_distance_validity(distance, direction, distance_sensors)

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
    valid = check_direction_distance_validity(distance, direction, distance_sensors)
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


def augment_data_network_heuristic(adjacency_network) -> any:
    initial_setup()
    global storage

    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    adjacency_network = adjacency_network.to(get_device())
    adjacency_network.eval()

    distance_metric = build_find_adjacency_heursitic_adjacency_network(adjacency_network)
    new_connections = evaluate_distance_metric(storage, distance_metric, storage.get_all_datapoints())

    # write_other_data_to_file(f"additional_found_connections_random_walks_300.json", new_connections)
    # storage.incorporate_new_data([], new_connections)

    return new_connections


def load_storage_with_base_data(storage):
    random_walk_datapoints = []
    random_walk_connections = []

    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    return storage


def augment_storage_with_saved_connections(storage) -> any:
    initial_setup()

    new_connections1 = read_other_data_from_file(f"additional_found_connections_rawh_random_walks_300.json")
    flag_data_authenticity(new_connections1)
    storage.incorporate_new_data([], new_connections1)
    new_connections2 = read_other_data_from_file(f"additional_found_connections_networkh_random_walks_300_24rot.json")
    flag_data_authenticity(new_connections2)
    storage.incorporate_new_data([], new_connections2)
    return storage


def augment_data_raw_heuristic() -> any:
    initial_setup()
    global storage, seen_network, neighborhood_distance_network, autoencoder, SSDir_network, SDirDistState_network

    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_random_walks_300_24rot.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    distance_metric = build_find_adjacency_heursitic_raw_data(storage)
    new_connections = evaluate_distance_metric(storage, distance_metric, storage.get_all_datapoints())
    return new_connections


storage: StorageSuperset2 = None
seen_network: SeenNetwork = None
neighborhood_distance_network: NeighborhoodDistanceNetwork = None
autoencoder: AutoencoderExploration = None
SSDir_network: SSDirNetwork = None
SDirDistState_network: SDirDistState = None
