import time
from entry_test import visualize_datapoints_reconstructions
from src.ai.runtime_storages.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.data_augmentation import load_storage_with_base_data, \
    storage_augment_with_saved_connections, augment_saved_connections_with_distances, \
    storage_augment_with_saved_augmented_connections, get_augmented_connections, \
    storage_augment_with_saved_connections_already_augmented
from src.ai.variants.exploration.data_filtering import data_filtering_redundancies, data_filtering_redundant_connections
from src.ai.variants.exploration.exploration_autonomous_policy import exploration_policy_autonomous, SDirDistS_network
from src.ai.variants.exploration.inference_policy import teleportation_exploring_inference, \
    teleportation_exploring_inference_evaluator
from src.configs_setup import config_simulation_communication
from src.modules.agent_communication import start_server
import threading

from src.modules.agent_communication.action_detach import detach_robot_teleport_absolute
from src.modules.agent_communication.communication_controller import set_response_event


def start_server_thread():
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()
    print("server thread started")


def exploration_autonomous_pipeline():
    config_simulation_communication(set_response_event)
    start_server_thread()
    exploration_policy_autonomous()


def inference_pipeline():
    config_simulation_communication(set_response_event)

    storage = StorageSuperset2()
    load_storage_with_base_data(
        storage=storage,
        datapoints_filename="(1)_datapoints_autonomous_walk.json",
        connections_filename="(1)_connections_autonomous_walk_augmented_filled.json"
    )

    start_server_thread()

    teleportation_exploring_inference_evaluator(
        models_folder="manually_saved",
        manifold_encoder_name="manifold_network_normal",
        SDirDistS_name="sdirdiststate_network_0.001",
        storage_arg=storage,
        noise=True
    )


def test_pipeline():
    exploration_autonomous_pipeline()

    pass
