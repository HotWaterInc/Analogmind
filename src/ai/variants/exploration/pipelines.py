from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.data_augmentation import load_storage_with_base_data, \
    storage_augment_with_saved_connections, augment_saved_connections_with_distances, \
    storage_augment_with_saved_augmented_connections, get_augmented_connections
from src.ai.variants.exploration.data_filtering import data_filtering_redundancies, data_filtering_redundant_connections
from src.ai.variants.exploration.exploration_autonomous_policy import exploration_policy_autonomous
from src.ai.variants.exploration.inference_policy import teleportation_exploring_inference
from src.ai.variants.exploration.inferences import fill_augmented_connections_distances
from src.ai.variants.exploration.networks.SDirDistState_network import SDirDistState, train_Sdirdiststate
from src.ai.variants.exploration.networks.SSDir_network import SSDirNetwork, train_SSDirection
from src.ai.variants.exploration.networks.images_raw_distance_predictor import ImagesRawDistancePredictor, \
    train_images_raw_distance_predictor
from src.ai.variants.exploration.networks.manifold_network import run_manifold_network, ManifoldNetwork
from src.ai.variants.exploration.others.abstraction_block_second_trial import run_abstraction_block_second_trial, \
    AbstractionBlockSecondTrial
from src.ai.variants.exploration.others.images_distance_predictor import train_images_distance_predictor, \
    ImagesDistancePredictor
from src.ai.variants.exploration.temporary import augment_data_testing_network_distance
from src.ai.variants.exploration.utils import storage_to_manifold
from src.modules.external_communication import start_server
from src.configs_setup import configs_communication, config_data_collection_pipeline
import threading
from src.modules.policies.data_collection import grid_data_collection
from src.ai.variants.camera1_full_forced.policy_images_simple import get_closest_point_policy
from src.modules.save_load_handlers.ai_models_handle import save_ai_manually, load_manually_saved_ai
from src.modules.save_load_handlers.data_handle import write_other_data_to_file


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
    load_storage_with_base_data(storage)

    generator = teleportation_exploring_inference(
        models_folder="manually_saved",
        autoencoder_name="manifold_network_034_031",
        SSD_name="ssdir_network_02",
        SDirDistS_name="sdirdiststate_network_001",
        storage_arg=storage
    )

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def data_augmentation_pipeline():
    storage = StorageSuperset2()
    load_storage_with_base_data(storage)
    storage_augment_with_saved_connections(storage)


def test_pipeline():
    # inference_pipeline()
    storage = StorageSuperset2()
    load_storage_with_base_data(storage)

    # connections = get_augmented_connections()

    # fill_augmented_connections_distances(connections, storage, ssdir_network)
    # fill_augmented_connections_distances()

    # data_filtering_redundant_connections(storage)
    # storage.build_non_adjacent_numpy_array_from_connections(debug=True)

    pass
