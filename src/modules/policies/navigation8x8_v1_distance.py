import time
import math
from typing import Dict, TypedDict, Generator, List
from src.action_ai_controller import ActionAIController
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.save_load_handlers.data_handle import write_other_data_to_file

from src.action_robot_controller import detach_robot_sample_distance, detach_robot_sample_image, \
    detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute, \
    detach_robot_rotate_continuous_absolute, detach_robot_forward_continuous
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def testing():
    for i in range(36):
        degrees = i * 10
        radians = degrees_to_radians(degrees)
        trick = normal_radians_to_webots(radians)

        detach_robot_rotate_continuous_absolute(trick)
        yield
        detach_robot_sample()
        yield

        global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()

        buffer = global_data_buffer.buffer
        data = buffer["data"]
        angle = buffer["params"]["angle"]
        x, y = buffer["params"]["x"], buffer["params"]["y"]
        print(webots_radians_to_normal(angle), angle, degrees)


storage: StorageSuperset2 = None
permutor = None
autoencoder: BaseAutoencoderModel = None
direction_network_SSD = None

DIRECTION_NETWORK = "direction_noauto_w92.pth"
PERMUTOR_NETWORK = "permutor_deshift_working.pth"
AUTOENCODER_NETWORK = "autoencodPerm10k_working.pth"


def load_everything():
    global storage, permutor, autoencoder, direction_network_SSD

    permutor = load_manually_saved_ai(PERMUTOR_NETWORK)
    storage = StorageSuperset2()

    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")

    storage.freeze_non_normalized_data()

    storage.normalize_all_data_super()
    storage.tanh_all_data()
    storage.set_permutor(permutor)
    storage.build_permuted_data_raw_with_thetas()

    # autoencoder = load_manually_saved_ai(AUTOENCODER_NETWORK)
    direction_network = load_manually_saved_ai(DIRECTION_NETWORK)


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


def navigation8x8() -> Generator[None, None, None]:
    load_everything()
    global storage, permutor, autoencoder, direction_network_SSD
    permutor = permutor.to(device)
    # autoencoder = autoencoder.to(device)
    permutor.eval()
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
            detach_robot_sample()
            yield

            global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
            buffer = global_data_buffer.buffer
            data = buffer["data"]
            angle = buffer["params"]["angle"]
            angle = webots_radians_to_normal(angle)
            empty_global_data_buffer()

            # inverting radian angles because of bug
            # angle = 2 * math.pi - angle
            angle_percent = angle / (2 * math.pi)

            print("current angle", angle, angle_percent)

            # run data through permutorDeshifter
            data = storage.normalize_incoming_data(data)
            data = array_to_tensor(np.tanh(data)).unsqueeze(0).to(device)
            # standard theta length 32 for this permutor
            thetas = build_thetas(angle_percent, 36)
            thetas = array_to_tensor(thetas).unsqueeze(0).to(device)
            standardized_data = permutor(data, thetas)
            # run data through autoencoder
            current_embedding = standardized_data
            target_embedding = storage.get_datapoint_data_tensor_by_name(f"{i}_{j}")[0].unsqueeze(0).to(device)
            ab_step = target_embedding - current_embedding
            ab_step = ab_step / torch.norm(ab_step)

            next_embedding = current_embedding + ab_step
            distance = torch.norm(target_embedding.squeeze(0) - current_embedding.squeeze(0), p=2, dim=0).item()
            print("DISTANCE", distance)

            # if distance < 0.5:
            #     print("HAS FINISHED TARGET REACHED")
            #     target_reached = False
            #     break

            direction_network = direction_network.to(device)

            # print("Current embedding:", current_embedding)
            # print("Target embedding:", target_embedding)

            direction = direction_network(current_embedding, next_embedding).squeeze(0)
            # print("Direction:", direction)
            print("argmax", torch.argmax(direction).item())
            angle = angle_policy(direction)
            print("Angle:", angle)

            # add angle noise
            angle += np.random.normal(0, 0.1)

            detach_robot_rotate_absolute(angle)
            yield
            detach_robot_forward_continuous(0.1)
            yield

        # def test_angles_direction():
#     load_everything()
#     global storage, permutor, autoencoder, direction_network
#     permutor = permutor.to(device)
#     autoencoder = autoencoder.to(device)
#     direction_network = direction_network.to(device)
#
#     # test 10 samples
#     samples = [
#         [0, 0, 0, 1],
#         [0, 0, 1, 0],
#         [0, 0, 1, 1],
#         [2, 2, 3, 3],
#         [2, 2, 3, 2],
#         [2, 2, 2, 3],
#         [2, 2, 1, 1],
#         [2, 2, 1, 2],
#         [2, 2, 1, 3],
#         [2, 2, 2, 1]
#     ]
#
#     for sample in samples:
#         i, j, k, l = sample
#         print("i:", i, "j:", j, "k:", k, "l:", l)
#
#         start_name = f"{i}_{j}"
#         end_name = f"{k}_{l}"
#         start_data = storage.get_datapoint_data_tensor_by_name(start_name)[0].unsqueeze(0).to(device)
#         end_data = storage.get_datapoint_data_tensor_by_name(end_name)[0].unsqueeze(0).to(device)
#
#         angle = 0
#         angle_percent = angle / (2 * math.pi)
#
#         thetas = build_thetas(angle_percent, 36)
#         thetas = array_to_tensor(thetas).unsqueeze(0).to(device)
#
#         standardized_data = permutor(start_data, thetas)
#
#         current_embedding = autoencoder.encoder_inference(standardized_data).to(device)
#         target_embedding = storage.get_datapoint_data_tensor_by_name(f"{k}_{l}")[0].unsqueeze(0).to(device)
#
#         ab_step = target_embedding - current_embedding
#         ab_step = ab_step / torch.norm(ab_step)
#
#         next_embedding = current_embedding + ab_step * 2
#
#         # print("Current embedding:", current_embedding)
#         # print("Target embedding:", target_embedding)
#
#         direction = direction_network(current_embedding, next_embedding).squeeze(0)
#         direction2 = direction_network(current_embedding, target_embedding).squeeze(0)
#
#         print("Direction step:", direction)
#         print("Direction target:", direction2)
#
#         # angle = calculate_angle([0, 1], [direction[0].item(), direction[1].item()])
#         # print("Angle:", angle)
#         # detach_robot_rotate_continuous_absolute(angle)
#         # yield
