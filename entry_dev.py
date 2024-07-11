import sys
import time

from src.modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import configs
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror
from src.modules.external_communication.communication_interface import send_data, CommunicationInterface
from src.utils import get_instance
from src.action_robot_controller import detach_robot_sample, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def take_user_input():
    while True:
        user_input = input("Enter command: ")

        detach_robot_sample()


if __name__ == "__main__":
    """
    This entry is used to develop and test while ai models are still training in the background.
    """

    configs()

    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # take_user_input()
    server_thread.join()
