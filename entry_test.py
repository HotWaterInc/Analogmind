import sys
from src.modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import configs, configs_communication, config_data_collection_pipeline
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.permutor import run_permutor
from src.ai.models.permutor_deshift import run_permutor_deshift
from src.ai.models.variational_autoencoder import *
from src.utils import perror
from src.modules.external_communication.communication_interface import send_data, CommunicationInterface
from src.utils import get_instance
from src.action_robot_controller import detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute
from src.modules.policies.data_collection import grid_data_collection
from src.ai.models.permutor_autoenc_pipelined import run_permuted_autoencoder
from src.ai.models.permutor_autoenc_pipelined2 import run_permuted_autoencoder2
from src.ai.models.direction_network_final import run_direction_network
from src.ai.models.direction_network_final2 import run_direction_network2

from src.ai.models.direction_network_ensemble import run_direction_network_ensemble

from src.modules.policies.navigation8x8_v1_distance import navigation8x8

from src.ai.models.autoencoder_images_north import run_autoencoder_images_north
from src.ai.models.autoencoder_images_full_forced import run_autoencoder_images_full
from src.ai.models.autoencoder_images_stacked_thetas import run_autoencoder_images_stacked_thetas
from src.ai.models.autoencoder_images_north_ensemble import run_autoencoder_ensemble_north

from src.ai.models.direction_network_images_final import run_direction_network_images_final

import time
import asyncio
from src.modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import configs
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror
from src.modules.external_communication.communication_interface import send_data, CommunicationInterface
from src.utils import get_instance
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative
import torch
import math
from scipy.stats import norm
import torch
from torchvision import models, transforms
from PIL import Image
import time
from src.modules.save_load_handlers.data_handle import read_other_data_from_file, write_other_data_to_file

from src.ai.models.autoencoder_images_north import run_autoencoder_images_north
from src.ai.models.autoencoder_images_full_forced import run_autoencoder_images_full
from src.ai.models.autoencoder_images_abstract_split import run_autoencoder_abstraction_block_images
from src.ai.models.autoencoder_images_stacked_thetas import run_autoencoder_images_stacked_thetas
from src.ai.models.autoencoder_images_north_ensemble import run_autoencoder_ensemble_north

from src.ai.models.direction_network_images_final import run_direction_network_images_final
from src.modules.policies.navigation_image_rawdirect import navigation_image_rawdirect, test_images_accuracy


def build_resnet18_embeddings():
    # Check if CUDA is available and set the device
    json_data_ids = read_other_data_from_file("data5x5_rotated24_images_id.json")
    new_json_connections = []

    lng = len(json_data_ids)
    for i in range(lng - 1):
        for j in range(i + 1, lng):
            name_start = json_data_ids[i]["name"]
            name_end = json_data_ids[j]["name"]

            i_start = json_data_ids[i]["params"]["i"]
            j_start = json_data_ids[i]["params"]["j"]
            i_end = json_data_ids[j]["params"]["i"]
            j_end = json_data_ids[j]["params"]["j"]

            json_dt = {
                "start": name_start,
                "end": name_end,
                "distance": 1.0,
                "direction": [
                    i_end - i_start,
                    j_end - j_start
                ]
            }

            if abs(i_end - i_start) + abs(j_end - j_start) == 1:
                new_json_connections.append(json_dt)
            elif abs(i_end - i_start) > 1:
                print(f"broken at i_start: {i_start}, i_end: {i_end}")
                break

    write_other_data_to_file("data5x5_connections.json", new_json_connections)


def navigation_direction_raw_image():
    configs_communication()
    generator = test_images_accuracy()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


if __name__ == "__main__":
    navigation_direction_raw_image()
    pass
