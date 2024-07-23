from src.configs_setup import configs
from src.modules.external_communication import start_server
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror
from src.modules.external_communication import start_server, CommunicationInterface
from src.ai.models.permutor_autoenc_pipelined2 import run_permuted_autoencoder2
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
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute
from src.modules.policies.data_collection import grid_data_collection
from src.ai.models.permutor_autoenc_pipelined import run_permuted_autoencoder
from src.modules.policies.navigation8x8_v1_distance import navigation8x8
from src.modules.policies.directed_data_collection import directed_data_collection

from src.ai.models.permutor_autoenc_pipelined2 import run_permuted_autoencoder2

from src.ai.models.direction_network_final import run_direction_network
from src.ai.models.direction_network_final2 import run_direction_network2


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def data_collection_pipeline():
    """
    Pipeline for collecting data from the robots
    Binds the server, and uses a generator like policy which sends data and awaits for response to call next(gen)
    """
    configs_communication()

    generator = grid_data_collection(3, 3, 15, 0, 0.5, 24, type="image")

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def navigation8x8pipeline():
    configs_communication()
    generator = navigation8x8()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def directed_data_collection_pipeline():
    configs_communication()
    generator = directed_data_collection()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


if __name__ == "__main__":
    # directed_data_collection_pipeline()
    # navigation8x8pipeline()
    data_collection_pipeline()

    # run_autoencoder()
    # run_permutor()

    # run_visualization()
    # run_permuted_autoencoder2()
    # run_direction_network2()
    pass
