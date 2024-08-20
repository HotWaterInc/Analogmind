from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.mutations import build_missing_connections_with_cheating
from src.modules.external_communication import start_server
from src.configs_setup import configs_communication, config_data_collection_pipeline
import threading
from src.modules.policies.data_collection import grid_data_collection
from src.modules.policies.navigation8x8_v1_distance import navigation8x8
from src.ai.variants.camera1_full_forced.autoencoder_images_full_forced import run_autoencoder_images_full_forced
from src.ai.models.autoencoder_images_north import run_autoencoder_images_north
from src.ai.variants.camera1_full_forced.policy_images_simple import navigation_image_1camera_vae, \
    get_closest_point_policy
from src.ai.variants.camera1_full_forced.direction_network_SS import run_direction_post_autoencod_SS
from src.ai.variants.camera1_full_forced.direction_network_SDS import run_direction_post_autoencod_SDS
from src.ai.variants.camera1_full_forced.vae_abstract_block_image import run_vae_abstract_block
from src.ai.variants.exploration.exploration_policy import exploration_policy, exploration_policy_train_only, \
    exploration_policy_augment_data

from src.ai.variants.exploration.inference_policy import teleportation_exploring_inference
from src.modules.save_load_handlers.data_handle import read_other_data_from_file


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


def exploration_pipeline():
    configs_communication()
    generator = exploration_policy()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


def inference_north_pipeline():
    configs_communication()
    models_folder = "exploration_inference_v1_north"
    autoencoder_name = "autoencoder_exploration_saved.pth"
    SSD_name = "SSDir_network_saved.pth"
    SDirDistS_name = "SDirDistState_network_saved.pth"
    storage: StorageSuperset2 = StorageSuperset2()
    random_walk_datapoints = read_other_data_from_file(f"datapoints_random_walks_500.json")
    random_walk_connections = read_other_data_from_file(f"datapoints_connections_randon_walks_500.json")
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)
    build_missing_connections_with_cheating(storage, random_walk_datapoints, distance_threshold=0.35)

    generator = teleportation_exploring_inference(models_folder, autoencoder_name, SSD_name, SDirDistS_name, storage)

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


if __name__ == "__main__":
    # exploration_pipeline()
    exploration_policy_train_only()
    # exploration_policy_augment_data()
    # inference_north_pipeline()

    pass
