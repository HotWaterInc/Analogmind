import torch
import numpy as np
import torch
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.utils import array_to_tensor
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from typing import List
from src.modules.time_profiler import start_profiler, profiler_checkpoint, profiler_checkpoint_blank
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display

import time
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super, evaluate_reconstruction_error_super_fist_rotation
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display
import torch
import torch.nn as nn


def evaluation_position_rotation_embeddings(model: BaseAutoencoderModel, storage: StorageSuperset2):
    ITERATIONS = 100
    ROTATIONS_PER_FULL = 12
    OFFSETS_PER_DATAPOINT = 23
    TOTAL_ROTATIONS = 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    positional_same_position_loss = 0
    positional_different_position_loss = 0

    for iteration in range(ITERATIONS):
        adjacency = storage.sample_adjacent_datapoints_connections(1)
        # positions = storage.sample_n_random_datapoints(2)
        positions = [adjacency[0]["start"], adjacency[0]["end"]]

        dp1_selected_array = []
        dp2_selected_array = []

        RANDOM_SAMPLES = 10
        for j in range(RANDOM_SAMPLES):
            sample = storage.get_point_rotations_with_full_info_random_offset_concatenated(positions[0],
                                                                                           ROTATIONS_PER_FULL)
            dp1_selected_array.append(array_to_tensor(sample))
            sample = storage.get_point_rotations_with_full_info_random_offset_concatenated(positions[1],
                                                                                           ROTATIONS_PER_FULL)
            dp2_selected_array.append(array_to_tensor(sample))

        # evaluate positional encodings for the first datapoint

        dp1_selected = torch.stack(dp1_selected_array).to(device)
        dp2_selected = torch.stack(dp2_selected_array).to(device)

        # we use encoder training because we want both embeddings
        dp1_positional, dp1_rotational = model.encoder_training(dp1_selected)
        dp2_positional, dp2_rotational = model.encoder_training(dp2_selected)

        positional_same_position_loss += torch.cdist(dp1_positional, dp1_positional).mean().item()
        positional_different_position_loss += torch.cdist(dp1_positional, dp2_positional).mean().item()

    print("")
    print("Positional same position loss: ", positional_same_position_loss / ITERATIONS)
    print("Positional different position loss: ", positional_different_position_loss / ITERATIONS)


def evaluation_position_rotation_embeddings_img1(model: BaseAutoencoderModel, storage: StorageSuperset2):
    ITERATIONS = 100

    ROTATIONS_PER_FULL = 1
    OFFSETS_PER_DATAPOINT = 24
    TOTAL_ROTATIONS = 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    positional_same_position_loss = 0
    positional_different_position_loss = 0

    for iteration in range(ITERATIONS):
        adjacency = storage.sample_adjacent_datapoints_connections(1)
        # positions = storage.sample_n_random_datapoints(2)
        positions = [adjacency[0]["start"], adjacency[0]["end"]]

        dp1_selected_array = []
        dp2_selected_array = []

        RANDOM_SAMPLES = 10
        for j in range(RANDOM_SAMPLES):
            sample = storage.get_point_rotations_with_full_info_random_offset_concatenated(positions[0],
                                                                                           ROTATIONS_PER_FULL)
            dp1_selected_array.append(array_to_tensor(sample))
            sample = storage.get_point_rotations_with_full_info_random_offset_concatenated(positions[1],
                                                                                           ROTATIONS_PER_FULL)
            dp2_selected_array.append(array_to_tensor(sample))

        # evaluate positional encodings for the first datapoint

        dp1_selected = torch.stack(dp1_selected_array).to(device)
        dp2_selected = torch.stack(dp2_selected_array).to(device)

        # we use encoder training because we want both embeddings
        dp1_positional, dp1_rotational = model.encoder_training(dp1_selected)
        dp2_positional, dp2_rotational = model.encoder_training(dp2_selected)

        positional_same_position_loss += torch.cdist(dp1_positional, dp1_positional).mean().item()
        positional_different_position_loss += torch.cdist(dp1_positional, dp2_positional).mean().item()

    print("")
    print("Positional same position loss: ", positional_same_position_loss / ITERATIONS)
    print("Positional different position loss: ", positional_different_position_loss / ITERATIONS)


def evaluation_position_rotation_embeddings_img1_on_00(model: BaseAutoencoderModel, storage: StorageSuperset2):
    ITERATIONS = 1

    ROTATIONS_PER_FULL = 1
    OFFSETS_PER_DATAPOINT = 24
    TOTAL_ROTATIONS = 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    positional_same_position_loss = 0
    positional_different_position_loss = 0

    for iteration in range(ITERATIONS):
        adjacency = storage.sample_adjacent_datapoints_connections(1)
        # positions = storage.sample_n_random_datapoints(2)
        positions = ["0_0", "0_0"]

        dp1_selected_array = []
        dp2_selected_array = []

        RANDOM_SAMPLES = 10
        sample = storage.get_datapoint_data_tensor_by_name("0_0")
        print(sample[0][:5])
        dp1_selected = sample.to(device)
        for j in range(RANDOM_SAMPLES):
            # sample = storage.get_point_rotations_with_full_info_random_offset_concatenated(positions[0],
            #                                                                                ROTATIONS_PER_FULL)
            # dp1_selected_array.append(array_to_tensor(sample))
            sample = storage.get_point_rotations_with_full_info_random_offset_concatenated(positions[1],
                                                                                           ROTATIONS_PER_FULL)
            dp2_selected_array.append(array_to_tensor(sample))

        # evaluate positional encodings for the first datapoint

        # dp1_selected = torch.stack(dp1_selected_array).to(device)
        dp2_selected = torch.stack(dp2_selected_array).to(device)

        # we use encoder training because we want both embeddings
        dp1_positional, dp1_rotational = model.encoder_training(dp1_selected)
        dp2_positional, dp2_rotational = model.encoder_training(dp2_selected)

        positional_same_position_loss += torch.cdist(dp1_positional, dp1_positional).mean().item()
        positional_different_position_loss += torch.cdist(dp1_positional, dp2_positional).mean().item()

    print("")
    print("Positional same position loss: ", positional_same_position_loss / ITERATIONS)
    print("Positional different position loss: ", positional_different_position_loss / ITERATIONS)
