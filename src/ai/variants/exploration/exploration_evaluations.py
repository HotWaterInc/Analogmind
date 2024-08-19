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


def evaluate_distance_metric(storage: StorageSuperset2, metric, new_datapoints: List[any],
                             distance_threshold):
    """
    Evaluate new datapoints and old datapoints with the distance metric
    """
    check_min_distance()

    THRESHOLD = distance_threshold
    should_be_found = []
    should_not_be_found = []

    print("Started evaluating metric")
    # finds out what new datapoints should be found as adjacent
    for new_datapoint in new_datapoints:
        minimum_distance = check_min_distance(storage, new_datapoint)
        if minimum_distance < THRESHOLD:
            should_be_found.append(new_datapoint)
        else:
            should_not_be_found.append(new_datapoint)

    print("calculated min distances")

    # finds out datapoints by metric
    found_datapoints = []
    negative_datapoints = []

    set_pretty_display(len(new_datapoints), "Distance metric evaluation")
    pretty_display_start()
    for idx, new_datapoint in enumerate(new_datapoints):
        if metric(storage, new_datapoint) == 1:
            found_datapoints.append(new_datapoint)
        else:
            negative_datapoints.append(new_datapoint)

        if idx % 10 == 0:
            pretty_display(idx)

    pretty_display_reset()

    print("calculated metric results")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    false_positives_arr = []

    for found_datapoint in found_datapoints:
        if found_datapoint in should_be_found:
            true_positives += 1
        else:
            false_positives += 1
            false_positives_arr.append(found_datapoint)

    for negative_datapoint in negative_datapoints:
        if negative_datapoint in should_not_be_found:
            true_negatives += 1
        else:
            false_negatives += 1

    if len(found_datapoints) == 0:
        print("No found datapoints for this metric")
        return

    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"True negatives: {true_negatives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {true_positives / (true_positives + false_positives)}")
    print(f"Recall: {true_positives / (true_positives + false_negatives)}")
    print(
        f"Accuracy: {(true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)}")

    for false_positive in false_positives_arr:
        distance = check_min_distance(storage, false_positive)
        print("false positive", distance)
