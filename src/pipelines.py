import threading
from src.agent_communication import start_server
from src.agent_communication.communication_controller import set_response_event
from src.configs_setup import config_simulation_communication
from src.navigation_core.autonomous_exploration.exploration_by_metadata import exploration_by_metadata
from src import runtime_storages as storage
from src.navigation_core.data_loading import load_storage_with_base_data
from src.runtime_storages.storage_struct import StorageStruct


def start_server_thread():
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()
    print("server thread started")


def exploration_autonomous_pipeline():
    config_simulation_communication(set_response_event)
    start_server_thread()
    exploration_by_metadata()


def inference_pipeline():
    pass
    # config_simulation_communication(set_response_event)
    #
    # storage_struct = StorageSuperset2()
    # load_storage_with_base_data(
    #     storage_struct=storage_struct,
    #     datapoints_filename="(1)_datapoints_autonomous_walk.json",
    #     connections_filename="(1)_connections_autonomous_walk_augmented_filled.json"
    # )
    #
    # start_server_thread()
    #
    # teleportation_exploring_inference_evaluator(
    #     models_folder="manually_saved",
    #     manifold_encoder_name="manifold_network_normal",
    #     SDirDistS_name="sdirdiststate_network_0.001",
    #     storage_arg=storage_struct,
    #     noise=True
    # )


def visualization_pipeline():
    storage_struct = StorageStruct()
    step = 5
    load_storage_with_base_data(
        storage_struct=storage_struct,
        nodes_filename=f"step{step}_datapoints_walk.json",
        connections_authentic_filename=f"step{step}_connections_authentic_walk.json",
        connections_synthetic_filename=f"step{step}_connections_synthetic_walk.json",
        connections_null_filename=f"step{step}_connections_null_walk.json"
    )


def test_pipeline():
    # visualization_pipeline()
    exploration_autonomous_pipeline()

    pass
