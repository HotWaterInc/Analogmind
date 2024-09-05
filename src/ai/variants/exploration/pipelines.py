from entry_test import visualize_datapoints_reconstructions
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.data_augmentation import load_storage_with_base_data, \
    storage_augment_with_saved_connections, augment_saved_connections_with_distances, \
    storage_augment_with_saved_augmented_connections, get_augmented_connections, \
    storage_augment_with_saved_connections_already_augmented
from src.ai.variants.exploration.data_filtering import data_filtering_redundancies, data_filtering_redundant_connections
from src.ai.variants.exploration.exploration_autonomous_policy import exploration_policy_autonomous, SDirDistS_network
from src.ai.variants.exploration.inference_policy import teleportation_exploring_inference
from src.ai.variants.exploration.inferences import fill_augmented_connections_distances
from src.ai.variants.exploration.networks.SDirDistState_network import SDirDistState, train_SDirDistS_network
from src.ai.variants.exploration.networks.SSDir_network import SSDirNetwork, train_SSDirection
from src.ai.variants.exploration.networks.images_raw_distance_predictor import ImagesRawDistancePredictor, \
    train_images_raw_distance_predictor
from src.ai.variants.exploration.networks.manifold_network import train_manifold_network, ManifoldNetwork
from src.ai.variants.exploration.networks.manifold_network_binary import ManifoldNetworkBinary, \
    train_manifold_network_binary
from src.ai.variants.exploration.networks.seen_network import train_seen_network, SeenNetwork
from src.ai.variants.exploration.others.abstraction_block_second_trial import run_abstraction_block_second_trial, \
    AbstractionBlockSecondTrial
from src.ai.variants.exploration.others.images_distance_predictor import train_images_distance_predictor, \
    ImagesDistancePredictor
from src.ai.variants.exploration.temporary import augment_data_testing_network_distance
from src.ai.variants.exploration.utils import storage_to_manifold, storage_to_binary_data
from src.modules.external_communication import start_server
from src.configs_setup import configs_communication, config_data_collection_pipeline
import threading
from src.modules.policies.data_collection import grid_data_collection
from src.ai.variants.camera1_full_forced.policy_images_simple import get_closest_point_policy
from src.modules.save_load_handlers.ai_models_handle import save_ai_manually, load_manually_saved_ai
from src.modules.save_load_handlers.data_handle import write_other_data_to_file
from src.modules.visualizations.entry import visualization_collected_data_photo


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def data_collection_pipeline():
    """
    Pipeline for collecting data from the robots
    Binds the server, and uses a generator like policy which sends data and awaits for response to call next(gen)
    """
    configs_communication()

    generator = grid_data_collection(3, 3, 5, 0, 0.5, 24, type="image")

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def navigation_1camera_vae_pipeline():
    configs_communication()
    # generator = navigation_image_1camera_vae()
    generator = get_closest_point_policy()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def exploration_autonomous_pipeline():
    configs_communication()
    generator = exploration_policy_autonomous()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def inference_pipeline():
    configs_communication()

    storage = StorageSuperset2()
    load_storage_with_base_data(
        storage=storage,
        datapoints_filename="step47_datapoints_autonomous_walk.json",
        connections_filename="step47_connections_autonomous_walk_augmented_filled.json"
    )

    generator = teleportation_exploring_inference(
        models_folder="manually_saved",
        manifold_encoder_name="manifold_network_2048_1_0.03_0.03",
        SSD_name="ssdir_network_0.02",
        SDirDistS_name="sdirdiststate_network_0.001",
        storage_arg=storage
    )

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def test_pipeline():
    exploration_autonomous_pipeline()
    # inference_pipeline()

    # storage = StorageSuperset2()
    # load_storage_with_base_data(
    #     storage=storage,
    #     datapoints_filename="step47_datapoints_autonomous_walk.json",
    #     connections_filename="step47_connections_autonomous_walk_augmented_filled.json"
    # )
    # manifold_network = load_manually_saved_ai("manifold_network_2048_1_0.03_0.03")
    # storage_to_manifold(storage, manifold_network)
    #
    # sdirdists = SDirDistState()
    # sdirdists = train_SDirDistS_network(sdirdists, storage)
    #
    # save_ai_manually(
    #     model=sdirdists,
    #     name="sdirdiststate_network",
    # )

    pass
