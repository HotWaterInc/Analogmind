import time
import math
from typing import Dict, TypedDict, Generator, List
from src.action_ai_controller import ActionAIController
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.save_load_handlers.data_handle import write_other_data_to_file

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
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def get_resnet18_embedding(image_array):
    # Load pre-trained ResNet-18 model

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)

    # Remove the final fully-connected layer
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Set the model to evaluation mode
    model.eval()

    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor = preprocess(image_array)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Get the embedding
        embedding = model(input_batch)

    # Reshape the embedding to a 1D tensor of size 512
    embedding = embedding

    return embedding


def degrees_to_radians(degrees: float) -> float:
    return degrees * math.pi / 180


def radians_to_degrees(radians: float) -> float:
    return radians * 180 / math.pi


def webots_radians_to_normal(x: float) -> float:
    if x < 0:
        x += 2 * math.pi
    return x


def normal_radians_to_webots(x: float) -> float:
    if x > math.pi:
        x -= 2 * math.pi
    return x


storage: StorageSuperset2 = None
direction_network = None

DIRECTION_NETWORK_PATH = "direction_image_raw.pth"


def load_everything():
    global storage, direction_network
    storage = StorageSuperset2()
    grid_dataset = 5

    storage.load_raw_data_from_others(f"data{grid_dataset}x{grid_dataset}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_dataset}x{grid_dataset}_connections.json")
    direction_network = load_manually_saved_ai(DIRECTION_NETWORK_PATH)


def calculate_angle(x_vector, y_vector):
    dot_product = x_vector[0] * y_vector[0] + x_vector[1] * y_vector[1]
    determinant = x_vector[0] * y_vector[1] - x_vector[1] * y_vector[0]
    angle = math.atan2(determinant, dot_product)
    return angle


def arg_to_angle(arg):
    angle = 0
    if arg == 0:
        angle = 0
    elif arg == 1:
        angle = math.pi / 2
    elif arg == 2:
        angle = math.pi
    elif arg == 3:
        angle = 3 * math.pi / 2
    return angle


def angle_policy(direction):
    softmax = nn.Softmax(dim=0)
    direction = softmax(direction)

    arg = torch.argmax(direction).item()
    arg_second = torch.argsort(direction, descending=True)[1].item()

    angle1 = arg_to_angle(arg)
    angle2 = arg_to_angle(arg_second)

    arg1_val = direction[arg].item()
    arg2_val = direction[arg_second].item()

    arg1_percent = arg1_val / (arg1_val + arg2_val)
    arg2_percent = arg2_val / (arg1_val + arg2_val)

    angle = angle1 * arg1_percent + angle2 * arg2_percent
    return angle


def angle_policy_simple(direction):
    arg = torch.argmax(direction).item()
    angle = arg_to_angle(arg)
    return angle


def next_embedding_policy_ab(current_embedding, target_embedding):
    # unstable heuristic
    ab_step = target_embedding - current_embedding
    ab_step = ab_step / torch.norm(ab_step)
    next_embedding = current_embedding + ab_step

    return next_embedding


THRESHOLD = 23
prev_best_distance = 10000


def find_closest_known_position(current_embedding, theta_percent):
    global storage
    best_embedding_distance = 100000
    best_embedding_name = None
    grid_dataset = 5
    current_theta_percent = theta_percent

    theta_search_index_left = int(current_theta_percent * 24)
    theta_search_index_right = int(current_theta_percent * 24) + 1

    for i in range(grid_dataset):
        for j in range(grid_dataset):
            target_name = f"{i}_{j}"
            potential_emb_left = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_left].to(
                device)
            potential_emb_right = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_right].to(
                device)

            distance_left_embedding = torch.norm(potential_emb_left - current_embedding, p=2, dim=0).item()
            distance_right_embedding = torch.norm(potential_emb_right - current_embedding, p=2, dim=0).item()

            if distance_left_embedding < best_embedding_distance:
                best_embedding_distance = distance_left_embedding
                best_embedding_name = target_name

            if distance_right_embedding < best_embedding_distance:
                best_embedding_distance = distance_right_embedding
                best_embedding_name = target_name

    return best_embedding_name


def next_embedding_policy_search_closest(current_embedding, current_theta_percent, target_embedding_i,
                                         target_embedding_j):
    global THRESHOLD, prev_best_distance
    # print("target embedding", target_embedding_i, target_embedding_j)

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
    potential_emb_left = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_left].to(device)
    potential_emb_right = storage.get_datapoint_data_tensor_by_name(target_name)[theta_search_index_right].to(device)

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
            theta_search_index_left].to(device)
        potential_current_embedding_right = storage.get_datapoint_data_tensor_by_name(potential_current_embedding_name)[
            theta_search_index_right].to(device)
        current_distance = connection["distance"]

        distance_left_embedding = torch.norm(potential_current_embedding_left - current_embedding, p=2, dim=0).item()
        distance_right_embedding = torch.norm(potential_current_embedding_right - current_embedding, p=2, dim=0).item()

        # print(connection)
        # print("distances", distance_left_embedding, distance_right_embedding)

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


def test_images_accuracy():
    width = 3
    height = 3
    grid_size = 5
    total_rotations = 24
    i, j = 4, 4
    rotation = 0
    x, y = get_position(width, height, grid_size, i, j, 0, 0.5)
    angle = get_angle(total_rotations, rotation)
    time.sleep(0.25)
    detach_robot_teleport_absolute(x, y)
    yield
    time.sleep(0.25)
    detach_robot_rotate_absolute(angle)
    yield
    detach_robot_sample_image_inference()
    yield

    global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
    buffer = global_data_buffer.buffer
    # print("Buffer:", buffer)
    image_data = buffer["data"]
    empty_global_data_buffer()

    nd_array_data = np.array(image_data)
    angle = buffer["params"]["angle"]
    angle = webots_radians_to_normal(angle)

    current_embedding = get_resnet18_embedding(nd_array_data).squeeze(-1).squeeze(-1).to(device)
    print("Current embedding:", current_embedding.shape)

    # get embedding for i,j,rot from data json
    target_embedding = storage.get_datapoint_data_tensor_by_name(f"{i}_{j}")[rotation].unsqueeze(0).to(device)

    # compare current embedding with target embedding
    distance = torch.norm(target_embedding.squeeze(0) - current_embedding.squeeze(0), p=2, dim=0)
    print("DISTANCE", distance.shape)
    print("DISTANCE", distance.item())


def navigation_image_rawdirect() -> Generator[None, None, None]:
    load_everything()
    global storage, direction_network

    direction_network.to(device)
    direction_network.eval()

    target_reached = False

    while True:
        i, j = 0, 0
        # takes i j from user
        i = int(input("Enter i: "))
        j = int(input("Enter j: "))
        print("i:", i, "j:", j)

        while target_reached is False:
            time.sleep(0.5)
            detach_robot_sample_image_inference()
            yield

            global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
            buffer = global_data_buffer.buffer
            # print("Buffer:", buffer)
            image_data = buffer["data"]
            empty_global_data_buffer()

            nd_array_data = np.array(image_data)
            angle = buffer["params"]["angle"]
            angle = webots_radians_to_normal(angle)

            # inverting radian angles because of bug
            current_embedding = get_resnet18_embedding(nd_array_data).squeeze(-1).squeeze(-1).to(device)
            target_embedding = storage.get_datapoint_data_tensor_by_name(f"{i}_{j}")[0].unsqueeze(0).to(device)

            # print("Current embedding:", current_embedding.shape, target_embedding.shape)

            # next_embedding = next_embedding_policy_ab(current_embedding.squeeze(), target_embedding.squeeze())
            angle_percent = angle / (2 * math.pi)
            closest = find_closest_known_position(current_embedding.squeeze(), angle_percent)
            print(closest)
            continue
            # next_embedding = next_embedding_policy_search_closest(current_embedding.squeeze(), angle_percent, i, j)

            # distance = torch.norm(target_embedding.squeeze(0) - current_embedding.squeeze(0), p=2, dim=0)
            # print("DISTANCE", distance.shape)

            # if distance < 0.5:
            #     print("HAS FINISHED TARGET REACHED")
            #     target_reached = False
            #     break

            direction_network = direction_network.to(device)
            direction = direction_network(current_embedding, next_embedding).squeeze(0)

            # print("argmax", torch.argmax(direction).item())
            angle = angle_policy_simple(direction)
            print("Angle:", radians_to_degrees(angle))

            # add angle noise
            # angle += np.random.normal(0, 0.1)

            detach_robot_rotate_absolute(angle)
            yield
            detach_robot_forward_continuous(0.1)
            yield
