import sys
import time

from src.modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import configs, configs_communication, config_data_collection_pipeline
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror
from src.modules.external_communication.communication_interface import send_data, CommunicationInterface
from src.utils import get_instance
from src.action_robot_controller import detach_robot_sample, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute
from src.modules.policies.data_collection import grid_data_collection


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def placeholder():
    yield
    print("placeholder")
    yield
    print("placeholder")
    yield


def data_collection_pipeline():
    """
    This entry is used to develop and test while ai models are still training in the background.
    """
    configs_communication()

    generator = grid_data_collection(1, 1, 5, 0, 0, 1)
    # generator = placeholder()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    server_thread.join()


if __name__ == "__main__":
    data_collection_pipeline()
