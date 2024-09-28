import copy
from src.navigation_core import data_filtering_redundant_connections, \
    data_filtering_redundant_datapoints
from src.navigation_core import exploration_inference
from src.navigation_core import fill_augmented_connections_distances, \
    fill_augmented_connections_distances_cheating, fill_augmented_connections_directions_cheating
from src.navigation_core import build_find_adjacency_heursitic_raw_data, \
    build_find_adjacency_heursitic_adjacency_network, build_find_adjacency_heursitic_cheating
from src.navigation_core import SDirDistState, \
    train_SDirDistS_network_until_threshold
from src.navigation_core import SSDirNetwork, train_SSDirection_until_threshold
from src.navigation_core import AdjacencyDetector, \
    train_adjacency_network_until_threshold
from src.navigation_core import ImagesRawDistancePredictor, \
    train_images_raw_distance_predictor_until_threshold
from src.navigation_core import ROTATIONS, STEP_DISTANCE_NULL_CONNECTION, \
    STEP_DISTANCE_LOWER_BOUNDARY, STEP_DISTANCE_UPPER_BOUNDARY
from src.navigation_core import get_collected_data_image, get_collected_data_distances, \
    check_direction_distance_validity_north, storage_to_manifold, find_frontier_all_datapoint_and_direction
from src.navigation_core import get_direction_between_datapoints, flag_data_authenticity, \
    generate_dxdy
from src.modules.agent_communication.action_detach import detach_robot_teleport_absolute, \
    detach_robot_teleport_relative, detach_robot_sample_image, detach_robot_sample_distance, \
    detach_robot_rotate_absolute
from src.save_load_handlers.ai_models_handle import save_ai_manually
from src.modules.save_load_handlers.data_handle import write_other_data_to_file
import time
from src.ai.runtime_storages.storage_superset2 import *
from typing import List, Dict
from src.utils import get_device
from src.navigation_core import ManifoldNetwork, \
    train_manifold_network_until_thresholds
from src.navigation_core import evaluate_distance_metric, \
    evaluate_distance_metric_on_already_found_connections


def initial_setup():
    global storage_raw, storage_manifold
    global manifold_network, SSDir_network, SDirDistState_network, image_distance_network, adjacency_network

    storage_raw = StorageSuperset2()
    storage_manifold = StorageSuperset2()

    image_distance_network = ImagesRawDistancePredictor().to(get_device())
    manifold_network = ManifoldNetwork().to(get_device())
    SSDir_network = SSDirNetwork().to(get_device())
    SDirDistState_network = SDirDistState().to(get_device())
    adjacency_network = AdjacencyDetector().to(get_device())


def collect_data_generator():
    detach_robot_sample_image()


def collect_distance_data_generator_without_sleep():
    detach_robot_sample_distance()


def collect_distance_data_generator_with_sleep():
    time.sleep(0.05)
    detach_robot_sample_distance()


def collect_image_data_generator_without_sleep():
    detach_robot_sample_image()


def collect_image_data_generator_with_sleep():
    time.sleep(0.1)
    detach_robot_sample_image()


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

movement_distances = np.random.uniform(STEP_DISTANCE_LOWER_BOUNDARY, STEP_DISTANCE_UPPER_BOUNDARY, 500)
index = 0


def get_random_movement():
    distance = movement_distances[index]
    direction = np.random.uniform(0, 2 * math.pi)

    return distance, direction


def check_position_is_known_cheating(random_walk_datapoints):
    global storage_raw
    current_datapoint = random_walk_datapoints[-1]
    current_x = current_datapoint["params"]["x"]
    current_y = current_datapoint["params"]["y"]
    is_known = storage_raw.check_node_is_known_metadata([current_x, current_y])

    return is_known


def random_move_policy():
    """
    Constrained only by distance sensors
    """

    global walk_directions_stack
    # collects sensor data and finds a valid move direction
    detach_robot_sample_distance()

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
            dx, dy = generate_dxdy(angle, STEP_DISTANCE_NULL_CONNECTION)
            direction = [dx, dy]
            distance = STEP_DISTANCE_NULL_CONNECTION
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
        collect_image_data_generator_without_sleep()

        image_embedding, angle, coords = get_collected_data_image()
        data_arr.append(image_embedding.tolist())

        collect_distance_data_generator_without_sleep()
        distances, angle, coords = get_collected_data_distances()

        # sensors oriented in the direction of the robot, so we can only check the north assuming it is rotated in the direction we want
        distance = STEP_DISTANCE_NULL_CONNECTION
        direction = 0
        valid = check_direction_distance_validity_north(distance, direction, distances)

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
    collect_data_rotations_and_create_datapoint()

    datapoint = return_global_buffer1()
    null_connections_test = return_global_buffer2()

    random_walk_datapoints.append(datapoint)
    non_valid_connections = null_connections_to_raw_connections_data(datapoint["name"], null_connections_test)
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)
    random_walk_connections.extend(non_valid_connections)


def random_walk_policy(random_walk_datapoints, random_walk_connections):
    collect_image_data_generator_with_sleep()

    image_embedding, angle, coords = get_collected_data_image()
    name = f"{coords[0]:.3f}_{coords[1]:.3f}"
    angle_percent = angle / (2 * math.pi)

    datapoint = create_datapoint(name, image_embedding.tolist(), coords)
    random_walk_datapoints.append(datapoint)
    add_connections_to_last_datapoint(random_walk_datapoints, random_walk_connections)

    random_move_policy()


def phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps, skip_checks=0):
    # if not first_walk:
    #     max_steps /= 10

    for step in range(max_steps):
        collect_current_data_and_add_connections(random_walk_datapoints, random_walk_connections)

        if first_walk == False and skip_checks == 0:
            position_check = check_position_is_known_cheating(random_walk_datapoints)
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


def exploration_policy_autonomous_data_filtering(step: int):
    global storage_raw, first_walk

    random_walk_datapoints = []
    random_walk_connections = []

    print(f"EXPLORING RANDOM WALK ITER {step}")
    if first_walk:
        print("IT'S FIRST TIME !!")

    #  phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps=3)
    random_walk_datapoints = read_other_data_from_file("datapoints_random_walks_300_24rot.json")
    random_walk_connections = read_other_data_from_file("datapoints_connections_random_walks_300_24rot.json")

    first_walk = False
    flag_data_authenticity(random_walk_connections)
    storage_raw.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    random_walk_datapoints_names = [datapoint["name"] for datapoint in random_walk_datapoints]

    new_connnections = augment_data_cheating_heuristic(storage_raw, random_walk_datapoints_names)
    total_connections_found = []
    total_connections_found.extend(new_connnections)
    flag_data_authenticity(total_connections_found)

    total_connections_found = fill_augmented_connections_distances_cheating(total_connections_found, storage_raw)
    total_connections_found = fill_augmented_connections_directions_cheating(total_connections_found,
                                                                             storage_raw)
    storage_raw.incorporate_new_data([], total_connections_found)

    print("DATA PURGING")
    data_filtering_redundant_connections(storage_raw)
    data_filtering_redundant_datapoints(storage_raw)
    storage_raw.build_non_adjacent_distances_from_connections(debug=True)


def exploration_policy_autonomous_exploration_full(step: int):
    global storage_raw, first_walk, exploring
    global adjacency_network, image_distance_network, manifold_network, SSDir_network, SDirDistS_network

    random_walk_datapoints = []
    random_walk_connections = []

    print(f"EXPLORING RANDOM WALK ITER {step}")
    if first_walk:
        print("IT'S FIRST TIME !!")

    phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps=30, skip_checks=3)

    first_walk = False
    flag_data_authenticity(random_walk_connections)

    storage_raw.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    random_walk_datapoints_names = [datapoint["name"] for datapoint in random_walk_datapoints]

    adjacency_network = train_adjacency_network_until_threshold(
        adjacency_network=adjacency_network,
        storage=storage_raw
    )

    print("AUGMENTING DATA")
    new_connections_raw = augment_data_raw_heuristic(storage_raw, random_walk_datapoints_names)
    new_connections_adjacency_network = augment_data_network_heuristic(storage_raw, random_walk_datapoints_names,
                                                                       adjacency_network)

    total_connections_found = []
    total_connections_found.extend(new_connections_raw)
    total_connections_found.extend(new_connections_adjacency_network)
    flag_data_authenticity(total_connections_found)

    evaluate_distance_metric_on_already_found_connections(storage_raw, random_walk_datapoints_names,
                                                          total_connections_found)
    print("FINISHING AUGMENTING CONNECTIONS")
    print("TRAINING DISTANCES NETWORK")

    image_distance_network = train_images_raw_distance_predictor_until_threshold(
        image_distance_predictor_network=image_distance_network,
        storage=storage_raw
    )

    print("ADDING SYNTHETIC DISTANCES")
    total_new_connections_filled = fill_augmented_connections_distances(
        additional_connections=total_connections_found,
        storage=storage_raw,
        image_distance_network=image_distance_network)

    storage_raw.incorporate_new_data([], total_new_connections_filled)

    print("DATA PURGING")
    # data_filtering_redundant_connections(storage_raw, verbose=False)
    # datapoints purging not implemented properly yet
    storage_raw.build_non_adjacent_distances_from_connections(debug=False)

    train_manifold_network_until_thresholds(
        manifold_network=manifold_network,
        storage=storage_raw,
    )

    print("FINISHED TRAINING MANIFOLD NETWORK")
    print("CREATING MANIFOLD STORAGE")
    copy_storage(
        storage_to_copy=storage_raw,
        storage_to_copy_into=storage_manifold
    )

    storage_to_manifold(
        storage=storage_manifold,
        manifold_network=manifold_network
    )

    last_dp = random_walk_datapoints[-1]
    frontier_connection = find_frontier_all_datapoint_and_direction(
        storage=storage_raw,
        return_first=True,
        starting_point=last_dp["name"]
    )

    if frontier_connection is None:
        print("NO FRONTIER FOUND, EXPLORATION FINISHED")
        exploring = False
        return

    write_other_data_to_file(f"step{step}_datapoints_autonomous_walk.json", storage_raw.get_raw_environment_data())
    write_other_data_to_file(f"step{step}_connections_autonomous_walk_augmented_filled.json",
                             storage_raw.get_raw_connections_data())

    frontier_datapoint = frontier_connection["start"]
    frontier_direction = frontier_connection["direction"]

    print("TRAINING SSDIR NETWORK")
    train_SSDirection_until_threshold(
        SSDir_network=SSDir_network,
        storage=storage_manifold
    )
    print("TRAINING SDirDistState NETWORK")
    train_SDirDistS_network_until_threshold(
        SDirDistState_network=SDirDistState_network,
        storage=storage_manifold
    )

    exploration_inference(
        storage_arg=storage_manifold,
        manifold_network_arg=manifold_network,
        direction_network_SSD_arg=SSDir_network,
        network_SDirDistS_arg=SDirDistState_network,
        target_name=frontier_datapoint
    )

    detach_robot_teleport_relative(frontier_direction[0], frontier_direction[1])

    print("TELEPORTED TO RELATIVE")
    time.sleep(1)


def exploration_policy_autonomous_exploration_cheating(step: int):
    global storage_raw, first_walk, exploring

    random_walk_datapoints = []
    random_walk_connections = []

    print(f"EXPLORING RANDOM WALK ITER {step}")
    if first_walk:
        print("IT'S FIRST TIME !!")

    phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps=20, skip_checks=2)

    first_walk = False
    flag_data_authenticity(random_walk_connections)
    storage_raw.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    random_walk_datapoints_names = [datapoint["name"] for datapoint in random_walk_datapoints]

    new_connnections = augment_data_cheating_heuristic(storage_raw, random_walk_datapoints_names)
    total_connections_found = []
    total_connections_found.extend(new_connnections)
    flag_data_authenticity(total_connections_found)

    total_connections_found = fill_augmented_connections_distances_cheating(total_connections_found, storage_raw)
    total_connections_found = fill_augmented_connections_directions_cheating(total_connections_found,
                                                                             storage_raw)
    storage_raw.incorporate_new_data([], total_connections_found)

    print("DATA PURGING")
    data_filtering_redundant_connections(storage_raw, verbose=False)
    # data_filtering_redundant_datapoints(storage_raw, verbose=False)
    storage_raw.build_non_adjacent_distances_from_connections(debug=False)

    last_dp = random_walk_datapoints[-1]
    frontier_connection = find_frontier_all_datapoint_and_direction(
        storage=storage_raw,
        return_first=True,
        starting_point=last_dp["name"]
    )

    if frontier_connection is None:
        print("NO FRONTIER FOUND, EXPLORATION FINISHED")
        exploring = False
        return

    write_other_data_to_file(f"step{step}_datapoints_autonomous_walk.json", storage_raw.get_raw_environment_data())
    write_other_data_to_file(f"step{step}_connections_autonomous_walk_augmented_filled.json",
                             storage_raw.get_raw_connections_data())

    frontier_datapoint = frontier_connection["start"]
    frontier_direction = frontier_connection["direction"]

    # move to frontier
    coords = storage_raw.get_datapoint_metadata_coords(frontier_datapoint)
    detach_robot_teleport_absolute(coords[0], coords[1])

    print("TELEPORTED TO FRONTIER")
    time.sleep(1)
    detach_robot_teleport_relative(frontier_direction[0], frontier_direction[1])

    print("TELEPORTED TO RELATIVE")
    time.sleep(1)


def exploration_policy_autonomous_step(step: int, train_networks=True):
    global storage_raw, storage_manifold, first_walk
    global adjacency_network, image_distance_network, manifold_network, SSDir_network, SDirDistS_network

    random_walk_datapoints = []
    random_walk_connections = []

    print(f"EXPLORING RANDOM WALK ITER {step}")
    if first_walk:
        print("IT'S FIRST TIME !!")

    phase_explore(random_walk_datapoints, random_walk_connections, first_walk, max_steps=3)
    first_walk = False
    flag_data_authenticity(random_walk_connections)
    storage_raw.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    random_walk_datapoints_names = [datapoint["name"] for datapoint in random_walk_datapoints]

    print("FINISHED RANDOM WALK")

    print("TRAINING ADJACENCY NETWORK")
    if train_networks:
        adjacency_network = train_adjacency_network_until_threshold(
            adjacency_network=adjacency_network,
            storage=storage_raw
        )
    print("AUGMENTING DATA")
    new_connections_raw = augment_data_raw_heuristic(storage_raw, random_walk_datapoints_names)
    new_connections_adjacency_network = augment_data_network_heuristic(storage_raw, random_walk_datapoints_names,
                                                                       adjacency_network)

    total_connections_found = []
    total_connections_found.extend(new_connections_raw)
    total_connections_found.extend(new_connections_adjacency_network)
    flag_data_authenticity(total_connections_found)

    evaluate_distance_metric_on_already_found_connections(storage_raw, random_walk_datapoints_names,
                                                          total_connections_found)
    print("FINISHING AUGMENTING CONNECTIONS")
    print("TRAINING DISTANCES NETWORK")

    if train_networks:
        image_distance_network = train_images_raw_distance_predictor_until_threshold(
            image_distance_predictor_network=image_distance_network,
            storage=storage_raw
        )

    print("ADDING SYNTHETIC DISTANCES")
    total_new_connections_filled = fill_augmented_connections_distances(
        additional_connections=total_connections_found,
        storage=storage_raw,
        image_distance_network=image_distance_network)

    storage_raw.incorporate_new_data([], total_new_connections_filled)

    print("SKIPPING DATA PURGING")
    print("TRAINING NAVIGATION NETWORKS")
    if train_networks:
        train_manifold_network_until_thresholds(
            manifold_network=manifold_network,
            storage=storage_raw,
        )
    print("FINISHED TRAINING MANIFOLD NETWORK")
    print("CREATING MANIFOLD STORAGE")
    copy_storage(
        storage_to_copy=storage_raw,
        storage_to_copy_into=storage_manifold
    )
    storage_to_manifold(
        storage=storage_manifold,
        manifold_network=manifold_network
    )

    print("TRAINING SSDIR NETWORK")
    if train_networks:
        train_SSDirection_until_threshold(
            SSDir_network=SSDir_network,
            storage=storage_manifold
        )
    print("TRAINING SDirDistState NETWORK")
    if train_networks:
        train_SDirDistS_network_until_threshold(
            SDirDistState_network=SDirDistState_network,
            storage=storage_manifold
        )

    save_ai_manually(f"step{step}_image_distance_network_auto", image_distance_network)
    save_ai_manually(f"step{step}_adjacency_network_auto", adjacency_network)
    save_ai_manually(f"step{step}_manifold_network_auto", manifold_network)
    save_ai_manually(f"step{step}_SSDir_network_auto", SSDir_network)
    save_ai_manually(f"step{step}_SDirDistState_network_auto", SDirDistState_network)

    write_other_data_to_file(f"step{step}_datapoints_autonomous_walk.json", storage_raw.get_raw_environment_data())
    write_other_data_to_file(f"step{step}_connections_autonomous_walk_augmented_filled.json",
                             storage_raw.get_raw_connections_data())

    print("NETWORKS ARE READY FOR INFERENCE !!!")
    print("ALL DATA WAS SAVED")


def exploration_policy_autonomous():
    initial_setup()
    global storage_raw, storage_manifold, first_walk, exploring
    global adjacency_network, image_distance_network, manifold_network, SSDir_network, SDirDistS_network

    detach_robot_teleport_absolute(0, 0)

    exploring = True
    first_walk = True
    step = 0

    while exploring:
        step += 1

        #  exploration_policy_autonomous_step(step, train_networks=False)
        # exploration_policy_autonomous_data_filtering(step)
        exploration_policy_autonomous_exploration_cheating(step)
        # generator = exploration_policy_autonomous_exploration_full(step)


adjacency_network: AdjacencyDetector = None
image_distance_network: ImagesRawDistancePredictor = None

manifold_network: ManifoldNetwork = None
SSDir_network: SSDirNetwork = None
SDirDistS_network: SDirDistState = None

global_register1 = None
global_register2 = None

first_walk = True
exploring = True
