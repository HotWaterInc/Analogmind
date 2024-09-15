import time
import math
from typing import Dict, TypedDict, Generator, List
from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.params import STEP_DISTANCE, MAX_DISTANCE, DISTANCE_THETAS_SIZE, DIRECTION_THETAS_SIZE
from src.ai.variants.exploration.utils import get_collected_data_distances, check_direction_distance_validity_north, \
    adjust_distance_sensors_according_to_rotation, adjust_distance_sensors_according_to_rotation_duplicate, \
    storage_to_manifold
from src.ai.variants.exploration.utils_pure_functions import distance_percent_to_distance_thetas, \
    angle_percent_to_thetas_normalized_cached, direction_to_degrees_atan, degrees_to_percent, \
    direction_thetas_to_radians, generate_dxdy, calculate_manifold_distances, radians_to_percent, radians_to_degrees
from src.modules.save_load_handlers.data_handle import write_other_data_to_file
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai, load_custom_ai, load_other_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_storages.storage_superset2 import StorageSuperset2
from src.utils import array_to_tensor, get_device
from src.modules.policies.testing_image_data import test_images_accuracy, process_webots_image_to_embedding, \
    squeeze_out_resnet_output, webots_radians_to_normal
import torch


def load_everything(models_folder: str, manifold_encoder_name: str, SDS_name: str):
    global direction_network_SDirDistS, manifold_network

    manifold_network = load_custom_ai(manifold_encoder_name, models_folder)
    direction_network_SDirDistS = load_custom_ai(SDS_name, models_folder)

    manifold_network.eval()
    direction_network_SDirDistS.eval()

    manifold_network = manifold_network.to(get_device())
    direction_network_SDirDistS = direction_network_SDirDistS.to(get_device())


def next_embedding_policy_ab(current_embedding, target_embedding):
    # unstable heuristic
    ab_step = target_embedding - current_embedding
    ab_step = ab_step / torch.norm(ab_step)
    next_embedding = current_embedding + ab_step

    return next_embedding


def find_closest_known_position(current_embedding, theta_percent):
    global storage
    best_embedding_distance = 100000
    best_embedding_name = None
    grid_dataset = 5
    current_theta_percent = theta_percent

    theta_search_index_left = int(current_theta_percent * 24)
    theta_search_index_right = int(current_theta_percent * 24) + 1
    if theta_search_index_right == 24:
        theta_search_index_right = 0

    datapoints = storage.get_all_datapoints()
    for datapoint in datapoints:
        target_name = datapoint
        potential_emb_left = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_left].to(
            get_device())
        potential_emb_right = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_right].to(
            get_device())

        distance_left_embedding = torch.norm(potential_emb_left - current_embedding, p=2, dim=0).item()
        distance_right_embedding = torch.norm(potential_emb_right - current_embedding, p=2, dim=0).item()

        if distance_left_embedding < best_embedding_distance:
            best_embedding_distance = distance_left_embedding
            best_embedding_name = target_name

        if distance_right_embedding < best_embedding_distance:
            best_embedding_distance = distance_right_embedding
            best_embedding_name = target_name

    return best_embedding_name


def find_closest_known_position_to_manifold_north(current_embedding, angle_percent):
    global storage
    best_embedding_distance = 100000
    best_embedding_name = None

    datapoints = storage.get_all_datapoints()
    for datapoint in datapoints:
        datapoint_name = datapoint

        potential_emb_left = storage.get_datapoint_data_tensor_by_name(datapoint_name)[0].to(
            get_device())
        distance_left_embedding = torch.norm(potential_emb_left - current_embedding, p=2, dim=0).item()

        if distance_left_embedding < best_embedding_distance:
            best_embedding_distance = distance_left_embedding
            best_embedding_name = datapoint_name

    return best_embedding_name


def next_embedding_policy_search_closest(current_embedding, current_theta_percent, target_embedding_i,
                                         target_embedding_j):
    # global THRESHOLD, prev_best_distance
    THRESHOLD = 0.5
    prev_best_distance = 100000

    global storage
    # searches the closest embedding to current embedding at a minimum distance from target embedding ( distance recorded wise )

    all_connections = storage.get_all_adjacent_data()
    target_name = f"{target_embedding_i}_{target_embedding_j}"
    # assumes 24 rotations in data

    theta_search_index_left = int(current_theta_percent * 24)
    theta_search_index_right = int(current_theta_percent * 24) + 1

    potential_current_embedding = None
    best_distance = 100000

    bestij = None

    # try for target position
    potential_emb_left = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_left].to(
        get_device())
    potential_emb_right = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_right].to(
        get_device())

    distance_left_embedding = torch.norm(potential_emb_left - current_embedding, p=2, dim=0).item()
    distance_right_embedding = torch.norm(potential_emb_right - current_embedding, p=2, dim=0).item()

    if distance_left_embedding < THRESHOLD:
        potential_current_embedding = potential_emb_left
        best_distance = 0
        bestij = target_name
    if distance_right_embedding < THRESHOLD:
        potential_current_embedding = potential_emb_right
        best_distance = 0
        bestij = target_name

    for connection in all_connections:
        potential_current_embedding_name = None

        if connection["start"] == target_name:
            potential_current_embedding_name = connection["end"]
        elif connection["end"] == target_name:
            potential_current_embedding_name = connection["start"]

        if potential_current_embedding_name is None:
            continue

        potential_current_embedding_left = storage.get_datapoint_data_tensor_by_name(potential_current_embedding_name)[
            theta_search_index_left].to(get_device())
        potential_current_embedding_right = storage.get_datapoint_data_tensor_by_name(potential_current_embedding_name)[
            theta_search_index_right].to(get_device())
        current_distance = connection["distance"]

        distance_left_embedding = torch.norm(potential_current_embedding_left - current_embedding, p=2, dim=0).item()
        distance_right_embedding = torch.norm(potential_current_embedding_right - current_embedding, p=2, dim=0).item()

        if current_distance <= best_distance and current_distance <= prev_best_distance:
            found_sol = False

            if distance_left_embedding < THRESHOLD:
                potential_current_embedding = potential_current_embedding_left
                found_sol = True

            if distance_right_embedding < THRESHOLD:
                potential_current_embedding = potential_current_embedding_right
                found_sol = True

            if distance_left_embedding < THRESHOLD and distance_right_embedding < THRESHOLD:
                found_sol = True
                if distance_left_embedding < distance_right_embedding:
                    potential_current_embedding = potential_current_embedding_left
                else:
                    potential_current_embedding = potential_current_embedding_right

            if found_sol:
                bestij = potential_current_embedding_name
                best_distance = current_distance
                prev_best_distance = best_distance

    if potential_current_embedding is None:
        print("NO POTENTIAL CURRENT EMBEDDING FOUND, INCREASING THRESHOLD")
        THRESHOLD += 1
        potential_current_embedding = current_embedding
    else:
        print("BESTIJ", bestij, "BEST DISTANCE", best_distance, "THRESHOLD", THRESHOLD)

    return potential_current_embedding


def print_closest_known_position(current_embedding, angle_percent):
    closest = find_closest_known_position(current_embedding, angle_percent)
    print("Closest known position:", closest)


def calculate_positions_manifold_distance(current_name: str, target_name: str,
                                          manifold_network: BaseAutoencoderModel, storage: StorageSuperset2):
    manifold_network.eval()
    manifold_network = manifold_network.to(get_device())

    current_embeddings = storage.get_datapoint_data_tensor_by_name(current_name).to(get_device())
    target_embeddings = storage.get_datapoint_data_tensor_by_name(target_name).to(get_device())

    current_manifold = manifold_network.encoder_inference(current_embeddings).mean(dim=0)
    target_manifold = manifold_network.encoder_inference(target_embeddings).mean(dim=0)

    distance = torch.norm(current_manifold - target_manifold, p=2, dim=0).item()
    return distance


def final_angle_policy_direction_testing(current_embedding, angle_percent, target_x, target_y, distance_sensors):
    global storage, direction_network_SDirDistS

    current_manifold = manifold_network.encoder_inference(current_embedding.unsqueeze(0)).squeeze()
    target_name = storage.get_closest_datapoint_to_xy(target_x, target_y)
    target_manifold = storage.get_datapoint_data_tensor_by_name(target_name)[0].to(get_device())

    closest = find_closest_known_position(current_manifold, angle_percent)
    # if closest == target_name:
    #     print("target reached")
    #     return None

    directions = []
    distances = []
    directions_radians = []

    previous_best_distance = calculate_manifold_distances(target_manifold, current_manifold)

    for angle in range(0, 360, 15):
        directions.append(generate_dxdy(math.radians(angle), STEP_DISTANCE))
        directions_radians.append(math.radians(angle))
        adjusting_factor = 0.5 if previous_best_distance < 0.1 else 1
        distances.append(STEP_DISTANCE * adjusting_factor)

    directions_percent = [degrees_to_percent(direction_to_degrees_atan(direction)) for direction in directions]
    directions_thetas = [angle_percent_to_thetas_normalized_cached(direction, DIRECTION_THETAS_SIZE) for direction in
                         directions_percent]
    distance_thetas = [distance_percent_to_distance_thetas(dist / MAX_DISTANCE, DISTANCE_THETAS_SIZE) for dist in
                       distances]

    directions_thetas = torch.stack(directions_thetas).to(get_device())
    distance_thetas = torch.stack(distance_thetas).to(get_device())
    predicted_manifolds = [
        direction_network_SDirDistS(current_manifold.unsqueeze(0), directions_thetas[idx].unsqueeze(0),
                                    distance_thetas[idx].unsqueeze(0)).squeeze(0)
        for
        idx, direction in enumerate(directions_thetas)]

    best_distance = 100000
    best_direction = None
    best_next_manifold = None
    best_next_angle = None

    for i, predicted_manifold in enumerate(predicted_manifolds):
        # check if direction is valid
        if not check_direction_distance_validity_north(distances[i] * 1.5, directions_radians[i],
                                                       adjust_distance_sensors_according_to_rotation(distance_sensors,
                                                                                                     angle_percent)):
            continue

        distance = torch.norm(predicted_manifold - target_manifold, p=2, dim=0).item()
        if distance < best_distance:
            best_distance = distance
            best_next_manifold = predicted_manifold
            best_next_angle = directions_radians[i]

    # if the chosen manifold is further from the target, it means we reached a local minima
    next_best_distance = calculate_manifold_distances(target_manifold, best_next_manifold)

    if (next_best_distance > previous_best_distance and previous_best_distance < 0.05) or previous_best_distance < 0.02:
        print("target reached")
        return None

    final_angle = best_next_angle
    # final_angle = policy_thetas_navigation_next_manifold(current_manifold, best_next_manifold)
    return final_angle


def teleportation_exploring_inference(models_folder: str, manifold_encoder_name: str,
                                      SDirDistS_name: str,
                                      storage_arg: StorageSuperset2) -> \
        Generator[
            None, None, None]:
    global manifold_network, storage
    storage = storage_arg
    load_everything(models_folder, manifold_encoder_name, SDirDistS_name)
    storage_to_manifold(storage, manifold_network)

    target_reached = False

    detach_robot_teleport_absolute(0, 0)
    yield

    while True:
        x = float(input("Enter x: "))
        y = float(input("Enter y: "))
        print("Navigating to", x, y)
        target_reached = False

        while target_reached is False:
            time.sleep(0.25)
            detach_robot_sample_image_inference()
            yield

            global_data_buffer: AgentResponseDataBuffer = AgentResponseDataBuffer.get_instance()
            buffer = global_data_buffer.buffer
            image_data = buffer["data"]
            empty_global_data_buffer()

            nd_array_data = np.array(image_data)
            angle = buffer["params"]["angle"]
            angle = webots_radians_to_normal(angle)

            angle_percent = angle / (2 * math.pi)

            current_embedding = process_webots_image_to_embedding(nd_array_data).to(get_device())
            current_embedding = squeeze_out_resnet_output(current_embedding)

            detach_robot_sample_distance()
            yield
            distance_sensors, angle, coords = get_collected_data_distances()
            final_angle = final_angle_policy_direction_testing(current_embedding, angle_percent, x, y, distance_sensors)

            if final_angle is None:
                target_reached = True
                continue

            detach_robot_rotate_absolute(final_angle)
            yield

            dx, dy = generate_dxdy(final_angle, STEP_DISTANCE / 2)
            detach_robot_teleport_relative(dx, dy)
            yield


def teleportation_exploring_inference_evaluator(models_folder: str, manifold_encoder_name: str,
                                                SDirDistS_name: str,
                                                storage_arg: StorageSuperset2, noise: bool = False) -> \
        Generator[
            None, None, None]:
    global manifold_network, storage
    storage = storage_arg
    load_everything(models_folder, manifold_encoder_name, SDirDistS_name)
    storage_to_manifold(storage, manifold_network)

    target_reached = False
    detach_robot_teleport_absolute(0, 0)
    yield

    x_min, y_min = -2, -1
    x_max, y_max = 1.5, 3.5
    TOTAL_TRIALS = 50
    STEPS = 50

    FAIL = 0
    WIN = 0

    STEPS_TO_WIN = []
    DISTANCE_FROM_TRUE_TARGET = []
    WINS = []
    RECORDED_STEPS = []
    ORIGINAL_TARGETS = []
    ACTUAL_POSITIONS = []

    for trial in range(TOTAL_TRIALS):
        x = x_min + (x_max - x_min) * np.random.rand()
        y = y_min + (y_max - y_min) * np.random.rand()
        target_reached = False
        steps_count = 0

        x_c = 0
        y_c = 0
        RECORDED_STEPS.append([])
        ACTUAL_POSITIONS.append([])

        while target_reached is False:
            time.sleep(0.25)
            detach_robot_sample_image_inference()
            yield

            global_data_buffer: AgentResponseDataBuffer = AgentResponseDataBuffer.get_instance()
            buffer = global_data_buffer.buffer
            image_data = buffer["data"]
            empty_global_data_buffer()

            nd_array_data = np.array(image_data)
            angle = buffer["params"]["angle"]
            angle = webots_radians_to_normal(angle)
            current_x, current_y = buffer["params"]["x"], buffer["params"]["y"]
            x_c, y_c = current_x, current_y

            angle_percent = angle / (2 * math.pi)

            current_embedding = process_webots_image_to_embedding(nd_array_data).to(get_device())
            current_embedding = squeeze_out_resnet_output(current_embedding)

            detach_robot_sample_distance()
            yield
            distance_sensors, angle, coords = get_collected_data_distances()
            final_angle = final_angle_policy_direction_testing(current_embedding, angle_percent, x, y, distance_sensors)

            ACTUAL_POSITIONS[-1].append((current_x, current_y))

            if final_angle is None:
                target_reached = True
                continue

            if noise:
                valid = False
                while not valid:
                    new_final_angle = final_angle + np.random.normal(0, 1)
                    new_final_angle = new_final_angle % (2 * math.pi)
                    if check_direction_distance_validity_north(STEP_DISTANCE / 2, new_final_angle,
                                                               adjust_distance_sensors_according_to_rotation(
                                                                   distance_sensors,
                                                                   angle_percent)):
                        valid = True
                        final_angle = new_final_angle

            detach_robot_rotate_absolute(final_angle)
            yield

            dx, dy = generate_dxdy(final_angle, STEP_DISTANCE / 2)
            detach_robot_teleport_relative(dx, dy)
            yield

            RECORDED_STEPS[-1].append((dx, dy))

            steps_count += 1
            if steps_count > STEPS:
                break

        target_name = storage.get_closest_datapoint_to_xy(x, y)
        true_coords = storage.get_datapoint_metadata_coords(target_name)
        x = true_coords[0]
        y = true_coords[1]
        ORIGINAL_TARGETS.append((x, y))

        STEPS_TO_WIN.append(steps_count)
        dist_target = math.sqrt((x_c - x) ** 2 + (y_c - y) ** 2)
        DISTANCE_FROM_TRUE_TARGET.append(dist_target)

        if dist_target < 0.35:
            WIN += 1
            WINS.append(1)
        else:
            FAIL += 1
            WINS.append(0)

        print("steps", steps_count, " dist_target", dist_target)

    print("TOTAL TRIALS", TOTAL_TRIALS)
    print("WIN", WIN)
    print("FAIL", FAIL)
    print("WINRATE", WIN / TOTAL_TRIALS)

    write_other_data_to_file(
        file_name="inference_policy_results.json",
        data={
            "STEPS_TO_WIN": STEPS_TO_WIN,
            "DISTANCE_FROM_TRUE_TARGET": DISTANCE_FROM_TRUE_TARGET,
            "WINS": WINS,
            "RECORDED_STEPS": RECORDED_STEPS,
            "ORIGINAL_TARGETS": ORIGINAL_TARGETS,
            "ACTUAL_POSITIONS": ACTUAL_POSITIONS
        }
    )


storage: StorageSuperset2 = None
direction_network_SDirDistS: nn.Module = None
manifold_network: BaseAutoencoderModel = None
