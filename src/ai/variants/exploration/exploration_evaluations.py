import time
import math
from typing import Dict, TypedDict, Generator, List, Tuple, Any
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.SDirDistState_network import run_SDirDistState, SDirDistState
from src.ai.variants.exploration.SSDir_network import run_SSDirection, SSDirNetwork, storage_to_manifold
from src.ai.variants.exploration.abstraction_block import run_abstraction_block_exploration, \
    AbstractionBlockImage
from src.ai.variants.exploration.evaluation_misc import run_tests_SSDir, run_tests_SSDir_unseen, \
    run_tests_SDirDistState
from src.ai.variants.exploration.mutations import build_missing_connections_with_cheating
from src.ai.variants.exploration.neighborhood_network import NeighborhoodDistanceNetwork, \
    run_neighborhood_network
from src.ai.variants.exploration.neighborhood_network_thetas import NeighborhoodNetworkThetas, \
    run_neighborhood_network_thetas, generate_new_ai_neighborhood_thetas, DISTANCE_THETAS_SIZE, MAX_DISTANCE
from src.ai.variants.exploration.seen_network import SeenNetwork, run_seen_network
from src.ai.variants.exploration.utils import get_collected_data_image, get_collected_data_distances, \
    evaluate_direction_distance_validity, check_min_distance
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.pretty_display import pretty_display_start, set_pretty_display, pretty_display
from src.modules.save_load_handlers.data_handle import write_other_data_to_file, serialize_object_other, \
    deserialize_object_other
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_sample_image, \
    detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute, \
    detach_robot_rotate_continuous_absolute, detach_robot_forward_continuous, detach_robot_sample_image_inference
import threading
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai, load_custom_ai, load_other_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import *
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super
from src.modules.policies.data_collection import get_position, get_angle
from src.modules.policies.testing_image_data import test_images_accuracy, process_webots_image_to_embedding, \
    squeeze_out_resnet_output
from src.modules.policies.utils_lib import webots_radians_to_normal, radians_to_degrees
import torch
from src.utils import get_device
from src.ai.variants.exploration.evaluation_exploration import print_distances_embeddings_inputs, \
    eval_distances_threshold_averages_seen_network
from src.ai.variants.exploration.autoencoder_network import run_autoencoder_network, AutoencoderExploration


def ground_truth_metric(storage: StorageSuperset2, new_datapoint: Dict[str, any], distance_threshold):
    """
    Find the closest datapoint in the storage
    """
    current_name = new_datapoint["name"]
    datapoints_names = storage.get_all_datapoints()
    adjacent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 1)
    adjacent_names.append(current_name)

    current_data_arr = []
    other_datapoints_data_arr = []
    selected_names = []

    for name in datapoints_names:
        if name in adjacent_names or name == current_name:
            continue

        selected_names.append(name)

    found_connections = []
    for name in selected_names:
        distance = storage.get_datapoints_real_distance(current_name, name)
        if distance < distance_threshold:
            found_connections.append(name)

    return found_connections


def evaluate_distance_metric(storage: StorageSuperset2, metric, new_datapoints: List[any]
                             ):
    """
    Evaluate new datapoints and old datapoints with the distance metric
    """
    true_positive = 0
    false_positive = 0
    really_bad_false_positive = 0

    new_datapoints = new_datapoints[:100]

    set_pretty_display(len(new_datapoints))
    pretty_display_start()

    all_found_datapoints = []
    true_found_datapoints = []

    for idx, new_datapoint in enumerate(new_datapoints):
        pretty_display(idx)

        found_datapoints = metric(storage, new_datapoint)
        all_found_datapoints.extend(found_datapoints)
        true_datapoints = ground_truth_metric(storage, new_datapoint, 0.5)
        true_found_datapoints.extend(true_datapoints)

        for founddp in found_datapoints:
            distance = storage.get_datapoints_real_distance(new_datapoint["name"], founddp)

            if distance < 0.5:
                true_positive += 1
            elif distance < 1:
                false_positive += 1
            else:
                really_bad_false_positive += 1

    print("")
    print("True positive", true_positive)
    print("Percent of true positive of actual positives", true_positive / len(true_found_datapoints))
    print("False positive", false_positive)
    print("Really bad false positive", really_bad_false_positive)
