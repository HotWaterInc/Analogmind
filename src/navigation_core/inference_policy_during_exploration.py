import math
from typing import Generator
from src.navigation_core import BaseAutoencoderModel
from src.navigation_core import STEP_DISTANCE, MAX_DISTANCE, DISTANCE_THETAS_SIZE, DIRECTION_THETAS_SIZE
from src.navigation_core import get_collected_data_distances, check_direction_distance_validity_north, \
    adjust_distance_sensors_according_to_rotation
from src.navigation_core import distance_percent_to_distance_thetas, \
    angle_percent_to_thetas_normalized_cached, direction_to_degrees_atan, degrees_to_percent, \
    direction_thetas_to_radians, generate_dxdy, calculate_manifold_distances
import time
import torch.nn as nn
import numpy as np
from src.ai.runtime_storages.storage_superset2 import StorageSuperset2
from src.utils import get_device
from src.modules.testing_image_data import process_webots_image_to_embedding, \
    squeeze_out_resnet_output, webots_radians_to_normal
import torch


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
    average_distance = 0

    for datapoint in datapoints:
        target_name = datapoint
        potential_emb_left = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_left].to(
            get_device())
        potential_emb_right = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_right].to(
            get_device())

        distance_left_embedding = torch.norm(potential_emb_left - current_embedding, p=2, dim=0).item()
        distance_right_embedding = torch.norm(potential_emb_right - current_embedding, p=2, dim=0).item()

        average_distance += min(distance_left_embedding, distance_right_embedding)

        if distance_left_embedding < best_embedding_distance:
            best_embedding_distance = distance_left_embedding
            best_embedding_name = target_name

        if distance_right_embedding < best_embedding_distance:
            best_embedding_distance = distance_right_embedding
            best_embedding_name = target_name

    average_distance /= len(datapoints)
    if best_embedding_distance > average_distance:
        return None

    return best_embedding_name


def final_angle_policy_direction_testing(current_embedding, angle_percent, target_name, distance_sensors):
    global storage, direction_network_SDirDistS

    current_manifold = manifold_network.encoder_inference(current_embedding.unsqueeze(0)).squeeze()
    target_manifold = storage.get_datapoint_data_tensor_by_name(target_name)[0].to(get_device())

    closest = find_closest_known_position(current_manifold, angle_percent)
    if closest == target_name or closest == None:
        print("target reached")
        if closest == None:
            print("closest is none!!!!")
        return None

    directions = []
    distances = []
    directions_radians = []

    for angle in range(0, 360, 15):
        directions.append(generate_dxdy(math.radians(angle), STEP_DISTANCE))
        directions_radians.append(math.radians(angle))
        distances.append(STEP_DISTANCE)

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

    previous_best_distance = calculate_manifold_distances(target_manifold, current_manifold)
    next_best_distance = calculate_manifold_distances(target_manifold, best_next_manifold)
    if next_best_distance > previous_best_distance:
        print("target reached")
        return None

    final_angle = policy_thetas_navigation_next_manifold(current_manifold, best_next_manifold)
    return final_angle


def policy_thetas_navigation_next_manifold(current_manifold: torch.Tensor, next_manifold: torch.Tensor):
    global direction_network_SSD

    direction_network_SSD = direction_network_SSD.to(get_device())
    direction_network_SSD.eval()

    thetas_direction = direction_network_SSD(current_manifold.unsqueeze(0), next_manifold.unsqueeze(0)).squeeze(0)
    final_angle = direction_thetas_to_radians(thetas_direction)
    return final_angle


def exploration_inference(storage_arg: StorageSuperset2, manifold_network_arg, direction_network_SSD_arg,
                          network_SDirDistS_arg, target_name: str) -> \
        Generator[
            None, None, None]:
    global storage, direction_network_SDirDistS, direction_network_SSD, manifold_network

    storage = storage_arg
    direction_network_SSD = direction_network_SSD_arg
    direction_network_SDirDistS = network_SDirDistS_arg
    manifold_network = manifold_network_arg

    target_reached = False

    detach_robot_teleport_absolute(0, 0)
    yield

    while True:
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
            final_angle = final_angle_policy_direction_testing(current_embedding, angle_percent, target_name,
                                                               distance_sensors)

            if final_angle is None:
                target_reached = True
                continue

            detach_robot_rotate_absolute(final_angle)
            yield

            dx, dy = generate_dxdy(final_angle, STEP_DISTANCE / 2)
            detach_robot_teleport_relative(dx, dy)
            yield


storage: StorageSuperset2 = None
direction_network_SSD: nn.Module = None
direction_network_SDirDistS: nn.Module = None
manifold_network: BaseAutoencoderModel = None
