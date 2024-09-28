import copy
import time
import math
from typing import List
from src import runtime_storages as storage

from src.agent_communication.action_detach import detach_agent_sample_distances, detach_agent_teleport_relative, \
    detach_agent_rotate_absolute, detach_agent_sample_image
from src.navigation_core.autonomous_exploration.common import get_collected_data_distances, random_distance_generator, \
    random_direction_generator, check_direction_validity, get_collected_data_image
from src.navigation_core.autonomous_exploration.exploration_autonomous_policy import \
    collect_image_data_generator_without_sleep, create_datapoint_multiple_rotations
from src.navigation_core.autonomous_exploration.params import NORTH
from src.navigation_core.autonomous_exploration.registers import set_list_buffer, set_value_buffer, get_value_buffer, \
    get_list_buffer, get_and_reset_value_buffer, get_and_reset_list_buffer
from src.navigation_core.pure_functions import generate_dxdy, flag_data_authenticity
from src.navigation_core.to_refactor.params import ROTATIONS, INVALID_DIRECTION_THRESHOLD
from src.runtime_storages.storage_struct import StorageStruct
from src.runtime_storages.types import NodeAuthenticData, ConnectionNullData, ConnectionAuthenticData


def initial_setup():
    global storage_struct
    storage_struct = storage.create_storage()


def check_position_is_known(random_walk_datapoints: list):
    global storage_struct
    current_datapoint = random_walk_datapoints[-1]
    current_x = current_datapoint["params"]["x"]
    current_y = current_datapoint["params"]["y"]
    is_known = storage.check_node_is_known_metadata(storage_struct, [current_x, current_y])
    return is_known


def random_move_policy():
    """
    Moves the agent randomly, while making sure it doesn't bump into obstacles
    """
    detach_agent_sample_distances()
    distance_sensors, angle, coords = get_collected_data_distances()

    valid = False
    distance = random_distance_generator()
    direction = random_direction_generator()
    # keeps distance constant and tries multiple directions
    while not valid:
        direction = random_direction_generator()
        valid = check_direction_validity(distance, direction, distance_sensors)

    dx, dy = generate_dxdy(direction, distance)
    detach_agent_teleport_relative(dx, dy)


def add_connections_to_last_datapoint(walk_nodes: List[NodeAuthenticData],
                                      walk_connections_authentic: List[ConnectionAuthenticData]):
    last_datapoint = walk_nodes[-1]
    added_conn = 0

    if len(walk_nodes) >= 2:
        added_conn += 1
        prev_datapoint = walk_nodes[-2]
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
        walk_connections_authentic.append(connection)

    if len(walk_nodes) >= 3:
        added_conn += 1
        prev_datapoint = walk_nodes[-3]
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
        walk_connections_authentic.append(connection)


def relative_difference(a, b):
    return abs(a - b) / ((a + b) / 2)


def collect_node_data_actions() -> tuple[NodeAuthenticData, List[ConnectionNullData]]:
    """
    Rotates the agent at the current location to collect all data
    """
    rotation_step = 2 * math.pi / ROTATIONS
    data_array = []
    null_connections: List[ConnectionNullData] = []

    name = None
    params = None

    for k in range(ROTATIONS):
        angle = k * rotation_step
        detach_agent_sample_image()

        detach_agent_rotate_absolute(angle)
        detach_agent_sample_image()

        image_embedding, angle, coords = get_collected_data_image()
        data_array.append(image_embedding.tolist())

        detach_agent_sample_distances()
        distances, angle, coords = get_collected_data_distances()

        if k == 0:
            name = f"{coords[0]:.3f}_{coords[1]:.3f}"
            params = {
                "x": coords[0],
                "y": coords[1]
            }

        # sensors oriented in the direction of the robot, so we can only check the north assuming it is rotated in the direction we want
        valid = check_direction_validity(INVALID_DIRECTION_THRESHOLD, NORTH, distances)

        if not valid:
            null_connection = ConnectionNullData(
                name=name,
                start=name,
                direction=NORTH,
                distance=INVALID_DIRECTION_THRESHOLD
            )
            null_connections.append(null_connection)

    datapoint = NodeAuthenticData(name=name, datapoints_array=data_array, params=params)

    return datapoint, null_connections


def collect_node_data(walk_nodes: List[NodeAuthenticData], walk_connections_authentic: List[ConnectionAuthenticData],
                      walk_connections_null: List[ConnectionNullData]):
    datapoint, null_connections = collect_node_data_actions()

    authentic_connections = add_connections_to_last_datapoint(walk_nodes, walk_connections_authentic)

    walk_connections_authentic.extend(null_connections)
    walk_nodes.append(datapoint)


def phase_explore(random_walk_datapoints: list, random_walk_connections: list, max_steps: int, skip_checks: int = 0):
    for step in range(max_steps):
        collect_node_data(random_walk_datapoints, random_walk_connections)

        if skip_checks == 0:
            position_check = check_position_is_known(random_walk_datapoints)
            if position_check:
                break

        if skip_checks != 0:
            skip_checks -= 1

        if step != max_steps - 1:
            random_move_policy()


def copy_storage(storage_to_copy: StorageSuperset2, storage_to_copy_into: StorageSuperset2):
    storage_to_copy_into = copy.deepcopy(storage_to_copy)
    return storage_to_copy_into


def augment_data_cheating_heuristic(storage: StorageSuperset2, random_walk_datapoints) -> any:
    distance_metric = build_find_adjacency_heursitic_cheating()
    new_connections = evaluate_distance_metric(storage, distance_metric, random_walk_datapoints)
    return new_connections


def augment_data_cheating_heuristic_simple(storage: StorageSuperset2) -> any:
    all_datapoints = storage.get_all_datapoints()
    new_connections = []

    for idx_i, datapoint_i in enumerate(all_datapoints):
        for idx_j, datapoint_j in enumerate(all_datapoints):
            if idx_i == idx_j:
                continue
            distance = get_real_distance_between_datapoints(datapoint_i, datapoint_j)
            direction = get_direction_between_datapoints(datapoint_i, datapoint_j)
            connection = {
                "start": datapoint_i["name"],
                "end": datapoint_j["name"],
                "distance": distance,
                "direction": direction
            }
            storage.add_connection(connection)

    return new_connections


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


def exploration_policy_autonomous_exploration_cheating(step: int):
    global storage_struct, first_walk, exploring

    random_walk_datapoints = []
    random_walk_connections = []

    print(f"EXPLORING RANDOM WALK ITER {step}")
    if first_walk:
        print("IT'S FIRST TIME !!")

    phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps=20, skip_checks=2)

    first_walk = False
    flag_data_authenticity(random_walk_connections)
    storage_struct.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    random_walk_datapoints_names = [datapoint["name"] for datapoint in random_walk_datapoints]

    new_connnections = augment_data_cheating_heuristic(storage_struct, random_walk_datapoints_names)
    total_connections_found = []
    total_connections_found.extend(new_connnections)
    flag_data_authenticity(total_connections_found)

    total_connections_found = fill_augmented_connections_distances_cheating(total_connections_found, storage_struct)
    total_connections_found = fill_augmented_connections_directions_cheating(total_connections_found,
                                                                             storage_struct)
    storage_struct.incorporate_new_data([], total_connections_found)

    print("DATA PURGING")
    data_filtering_redundant_connections(storage_struct, verbose=False)
    # data_filtering_redundant_datapoints(storage_raw, verbose=False)
    storage_struct.build_non_adjacent_distances_from_connections(debug=False)

    last_dp = random_walk_datapoints[-1]
    frontier_connection = find_frontier_all_datapoint_and_direction(
        storage=storage_struct,
        return_first=True,
        starting_point=last_dp["name"]
    )

    if frontier_connection is None:
        print("NO FRONTIER FOUND, EXPLORATION FINISHED")
        exploring = False
        return

    write_other_data_to_file(f"step{step}_datapoints_autonomous_walk.json", storage_struct.get_raw_environment_data())
    write_other_data_to_file(f"step{step}_connections_autonomous_walk_augmented_filled.json",
                             storage_struct.get_raw_connections_data())

    frontier_datapoint = frontier_connection["start"]
    frontier_direction = frontier_connection["direction"]

    # move to frontier
    coords = storage_struct.get_datapoint_metadata_coords(frontier_datapoint)
    detach_robot_teleport_absolute(coords[0], coords[1])

    print("TELEPORTED TO FRONTIER")
    time.sleep(1)
    detach_agent_teleport_relative(frontier_direction[0], frontier_direction[1])

    print("TELEPORTED TO RELATIVE")
    time.sleep(1)


def exploration_policy_autonomous():
    initial_setup()
    global storage_struct, storage_manifold, first_walk, exploring
    global adjacency_network, image_distance_network, manifold_network, SSDir_network, SDirDistS_network

    detach_robot_teleport_absolute(0, 0)

    exploring = True
    first_walk = True
    step = 0

    while exploring:
        step += 1

        exploration_policy_autonomous_exploration_cheating(step)


storage_struct: StorageStruct
