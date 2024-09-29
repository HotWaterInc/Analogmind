import time
import math
from typing import List
from src import runtime_storages as storage

from src.agent_communication.action_detach import detach_agent_sample_distances, detach_agent_teleport_relative, \
    detach_agent_rotate_absolute, detach_agent_sample_image, detach_agent_teleport_absolute
from src.navigation_core.autonomous_exploration.common import get_collected_data_distances, random_distance_generator, \
    random_direction_generator, check_direction_validity, get_collected_data_image
from src.navigation_core.autonomous_exploration.data_filtering import filtering_redundant_connections
from src.navigation_core.autonomous_exploration.metrics.functions import build_augmented_connections
from src.navigation_core.autonomous_exploration.params import NORTH
from src.navigation_core.pure_functions import generate_dxdy, get_real_distance_between_datapoints, \
    get_direction_between_datapoints
from src.navigation_core.autonomous_exploration.metrics.metric_builders import build_find_adjacency_heuristic_cheating
from src.navigation_core.autonomous_exploration.others import \
    synthetic_connections_fill_distances, synthetic_connections_fill_directions
from src.navigation_core.to_refactor.params import ROTATIONS, INVALID_DIRECTION_THRESHOLD, STEP_DISTANCE
from src.navigation_core.utils import find_frontier_all_datapoint_and_direction
from src.runtime_storages.storage_struct import StorageStruct
from src.runtime_storages.types import NodeAuthenticData, ConnectionNullData, ConnectionAuthenticData, \
    ConnectionSyntheticData
from src.save_load_handlers.data_handle import write_other_data_to_file


def initial_setup():
    global storage_struct
    storage_struct = storage.create_storage()


def check_position_is_known(random_walk_datapoints: list):
    global storage_struct
    current_datapoint = random_walk_datapoints[-1]
    current_x = current_datapoint["params"]["x"]
    current_y = current_datapoint["params"]["y"]
    is_known = storage.check_node_is_known_from_metadata(storage_struct, [current_x, current_y])
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


def _add_connection_to_previous_node(walk_nodes: List[NodeAuthenticData],
                                     added_connections: List[ConnectionAuthenticData], previous_rank: int = 1):
    last_datapoint = walk_nodes[-1]
    prev_datapoint = walk_nodes[-(previous_rank + 1)]
    start_name = prev_datapoint["name"]
    end_name = last_datapoint["name"]

    distance = get_real_distance_between_datapoints(prev_datapoint, last_datapoint)
    direction = get_direction_between_datapoints(prev_datapoint, last_datapoint)

    connection = ConnectionAuthenticData(
        name=f"{start_name}_{end_name}",
        start=start_name,
        end=end_name,
        distance=distance,
        direction=direction
    )
    added_connections.append(connection)


def add_connections_to_last_datapoint(walk_nodes: List[NodeAuthenticData]):
    added_connections: List[ConnectionAuthenticData] = []
    if len(walk_nodes) >= 2:
        _add_connection_to_previous_node(walk_nodes, added_connections, previous_rank=1)

    if len(walk_nodes) >= 3:
        _add_connection_to_previous_node(walk_nodes, added_connections, previous_rank=2)

    return added_connections


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
            dx, dy = generate_dxdy(angle, STEP_DISTANCE)
            null_connection = ConnectionNullData(
                name=name,
                start=name,
                direction=[dx, dy],
                distance=INVALID_DIRECTION_THRESHOLD
            )
            null_connections.append(null_connection)

    datapoint = NodeAuthenticData(name=name, datapoints_array=data_array, params=params)

    return datapoint, null_connections


def collect_node_data(walk_nodes: List[NodeAuthenticData], walk_connections_authentic: List[ConnectionAuthenticData],
                      walk_connections_null: List[ConnectionNullData]):
    datapoint, null_connections = collect_node_data_actions()
    authentic_connections = add_connections_to_last_datapoint(walk_nodes)
    walk_connections_authentic.extend(null_connections)
    walk_nodes.append(datapoint)


def explore_environment_and_collect_data(walk_nodes: List[NodeAuthenticData],
                                         walk_connections_authentic: List[ConnectionAuthenticData],
                                         walk_connections_null: List[ConnectionNullData],
                                         max_steps: int, skip_checks: int = 0):
    for step in range(max_steps):
        datapoint, null_connections = collect_node_data_actions()
        authentic_connections = add_connections_to_last_datapoint(walk_nodes)

        walk_connections_authentic.extend(authentic_connections)
        walk_connections_null.extend(null_connections)
        walk_nodes.append(datapoint)

        if skip_checks == 0:
            x, y = datapoint["params"]["x"], datapoint["params"]["y"]
            is_known = storage.check_node_is_known_from_metadata(storage_struct, [x, y])
            if is_known:
                break

        if skip_checks != 0:
            skip_checks -= 1

        if step != max_steps - 1:
            random_move_policy()


def augment_connections_by_metadata(storage_struct: StorageStruct, walk_nodes: List[str]) -> any:
    build_find_adjacency_heuristic_cheating()
    distance_metric = build_find_adjacency_heuristic_cheating()
    new_connections = build_augmented_connections(storage_struct, distance_metric, walk_nodes)
    return new_connections


def exploration_policy_autonomous_exploration_cheating(step: int):
    global storage_struct

    walk_nodes: List[NodeAuthenticData] = []
    walk_connections_authentic: List[ConnectionAuthenticData] = []
    walk_connections_null: List[ConnectionNullData] = []

    explore_environment_and_collect_data(
        walk_nodes=walk_nodes,
        walk_connections_authentic=walk_connections_authentic,
        walk_connections_null=walk_connections_null,
        max_steps=15,
        skip_checks=2
    )

    storage.crud.create_nodes(
        storage=storage_struct,
        nodes=walk_nodes
    )
    storage.crud.create_connections_authentic(
        storage=storage_struct,
        new_connections=walk_connections_authentic
    )
    storage.crud.create_connections_null(
        storage=storage_struct,
        new_connections=walk_connections_null
    )

    synthetic_connections_found: List[ConnectionSyntheticData] = []

    walk_nodes_names = [node["name"] for node in walk_nodes]
    new_connections: List[ConnectionSyntheticData] = augment_connections_by_metadata(storage_struct, walk_nodes_names)
    synthetic_connections_found.extend(new_connections)
    synthetic_connections_found = synthetic_connections_fill_distances(synthetic_connections_found,
                                                                       storage_struct)
    synthetic_connections_found = synthetic_connections_fill_directions(synthetic_connections_found,
                                                                        storage_struct)
    storage.crud.create_connections_synthetic(
        storage=storage_struct,
        new_connections=synthetic_connections_found
    )

    filtering_redundant_connections(storage_struct, verbose=False)
    # data_filtering_redundant_datapoints(storage_raw, verbose=False)

    last_dp = walk_nodes[-1]
    frontier_connection = find_frontier_all_datapoint_and_direction(
        storage_struct=storage_struct,
        return_first=True,
        starting_point=last_dp["name"]
    )

    if frontier_connection is None:
        print("NO FRONTIER FOUND, EXPLORATION FINISHED")
        return

    write_other_data_to_file(f"step{step}_datapoints_walk.json", storage.nodes_get_all(storage_struct))
    write_other_data_to_file(f"step{step}_connections_authentic_walk.json",
                             storage.connections_authentic_get(storage_struct))
    write_other_data_to_file(f"step{step}_connections_synthetic_walk.json",
                             storage.connections_synthetic_get(storage_struct))
    write_other_data_to_file(f"step{step}_connections_null_walk.json",
                             storage.connections_null_get(storage_struct))

    frontier_datapoint = frontier_connection["start"]
    frontier_direction = frontier_connection["direction"]

    # move to frontier
    coords = storage.node_get_coords_metadata(storage_struct, frontier_datapoint)

    detach_agent_teleport_absolute(coords[0], coords[1])
    print("TELEPORTED TO FRONTIER")
    time.sleep(1)
    detach_agent_teleport_relative(frontier_direction[0], frontier_direction[1])
    print("TELEPORTED TO RELATIVE")
    time.sleep(1)


def exploration_by_metadata():
    initial_setup()
    detach_agent_teleport_absolute(0, 0)

    exploring = True
    step = 0

    while exploring:
        step += 1
        exploration_policy_autonomous_exploration_cheating(step)


storage_struct: StorageStruct
