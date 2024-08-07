from src.modules.external_communication import start_server
from src.configs_setup import configs_communication, config_data_collection_pipeline
import threading
from src.modules.policies.data_collection import grid_data_collection
from src.modules.policies.navigation8x8_v1_distance import navigation8x8
from src.ai.variants.camera1_full_forced.autoencoder_images_full_forced import run_autoencoder_images_full_forced
from src.ai.models.autoencoder_images_north import run_autoencoder_images_north
from src.ai.variants.camera1_full_forced.policy_images_simple import navigation_image_1camera_vae
from src.ai.variants.camera1_full_forced.direction_network_SS import run_direction_post_autoencod_SS


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
    generator = navigation_image_1camera_vae()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


if __name__ == "__main__":
    # run_autoencoder_images_full_forced()

    navigation_1camera_vae_pipeline()
    # run_direction_post_autoencod_SS()
    pass
