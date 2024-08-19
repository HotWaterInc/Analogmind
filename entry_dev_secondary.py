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
from src.ai.variants.exploration.exploration_policy import exploration_policy

from src.ai.variants.exploration.inference_policy import teleportation_exploring_inference
from src.modules.save_load_handlers.data_handle import read_other_data_from_file
from src.modules.visualizations import run_visualization

if __name__ == "__main__":
    run_visualization()

    pass
