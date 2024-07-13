import sys
import time

from src.modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import configs, configs_communication, config_data_collection_pipeline
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.permutor import run_permutor
from src.ai.models.variational_autoencoder import *
from src.utils import perror
from src.modules.external_communication.communication_interface import send_data, CommunicationInterface
from src.utils import get_instance
from src.action_robot_controller import detach_robot_sample, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute
from src.modules.policies.data_collection import grid_data_collection
from src.ai.models.permutor_autoenc_pipelined import run_permuted_autoencoder
from src.ai.models.permutor_autoenc_pipelined2 import run_permuted_autoencoder2


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def data_collection_pipeline():
    """
    Pipeline for collecting data from the robots
    Binds the server, and uses a generator like policy which sends data and awaits for response to call next(gen)
    """
    configs_communication()

    generator = grid_data_collection(2, 2, 8, 0, 0, 20)

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


if __name__ == "__main__":
    # run_permutor()
    # run_autoencoder()
    run_permuted_autoencoder2()
    # run_permuted_autoencoder()
    pass
