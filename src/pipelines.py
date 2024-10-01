import threading
import time

from src.agent_communication import start_server
from src.agent_communication.action_detach import detach_agent_teleport_absolute
from src.agent_communication.communication_controller import set_response_event
from src.configs_setup import config_simulation_communication
from src.navigation_core.autonomous_exploration.exploration_by_metadata import exploration_by_metadata
from src.navigation_core.autonomous_exploration.exploration_profiling import exploration_profiling
from src.navigation_core.data_loading import load_storage_with_base_data
from src.navigation_core.networks.metric_generator import create_metric_network, train_metric_generator_network
from src.runtime_storages.storage_struct import StorageStruct
from src.visualizations.visualizations_static import visualization_3d_target_surface, visualization_topological_graph
from src import runtime_storages as storage


def start_server_thread():
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()
    print("server thread started")


def pipeline_exploration_profiling():
    config_simulation_communication(set_response_event)
    start_server_thread()
    exploration_profiling()


def pipeline_exploration_autonomous():
    config_simulation_communication(set_response_event)
    start_server_thread()
    exploration_by_metadata()


def pipeline_navigation():
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


def pipeline_visualization_metric3D():
    storage_struct = StorageStruct()
    step = 10
    load_storage_with_base_data(
        storage_struct=storage_struct,
        nodes_filename=f"step{step}_datapoints_walk.json",
        connections_authentic_filename=f"step{step}_connections_authentic_walk.json",
        connections_synthetic_filename=f"step{step}_connections_synthetic_walk.json",
        connections_null_filename=f"step{step}_connections_null_walk.json"
    )
    # also get the metric network here and pass it
    visualization_3d_target_surface(storage_struct, metric_network=None)


def pipeline_visualization_topology():
    storage_struct = StorageStruct()
    step = 10
    load_storage_with_base_data(
        storage_struct=storage_struct,
        nodes_filename=f"step{step}_datapoints_walk.json",
        connections_authentic_filename=f"step{step}_connections_authentic_walk.json",
        connections_synthetic_filename=f"step{step}_connections_synthetic_walk.json",
        connections_null_filename=f"step{step}_connections_null_walk.json"
    )
    visualization_topological_graph(storage_struct)


def pipeline_train_metric_network():
    storage_struct = StorageStruct()
    step = 10
    load_storage_with_base_data(
        storage_struct=storage_struct,
        nodes_filename=f"step{step}_datapoints_walk.json",
        connections_authentic_filename=f"step{step}_connections_authentic_walk.json",
        connections_synthetic_filename=f"step{step}_connections_synthetic_walk.json",
        connections_null_filename=f"step{step}_connections_null_walk.json"
    )
    metric_network = create_metric_network()
    train_metric_generator_network(storage_struct=storage_struct, network=metric_network)
    pass


def test_pipeline():
    pipeline_exploration_profiling()

    pass
