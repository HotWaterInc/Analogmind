import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, set_pretty_display, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import angle_to_thetas, thetas_to_radians, \
    angle_percent_to_thetas_normalized, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm

THETAS_SIZE = 36


def run_tests_SSDir_unseen(direction_network, storage: StorageSuperset2):
    direction_network = direction_network.to(get_device())
    direction_network.eval()

    datapoints: List[str] = storage.get_all_datapoints()

    win = 0
    lose = 0
    ITERATIONS = 1

    error_arr = []
    for iter in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()
        for datapoint in datapoints:
            connections_to_point: List[
                RawConnectionData] = storage.get_datapoints_adjacent_at_degree_n_as_raw_connection_data(datapoint, 3)

            for j in range(len(connections_to_point)):
                start = connections_to_point[j]["start"]
                end = connections_to_point[j]["end"]
                direction = connections_to_point[j]["direction"]

                direction = torch.tensor(direction, dtype=torch.float32, device=get_device())
                l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
                direction = direction / l2_direction

                start_data = storage.get_datapoint_data_tensor_by_name_permuted(start).to(get_device())
                end_data = storage.get_datapoint_data_tensor_by_name_permuted(end).to(get_device())
                metadata = storage.get_pure_permuted_raw_env_metadata_array_rotation()
                index_start = storage.get_datapoint_index_by_name(start)
                index_end = storage.get_datapoint_index_by_name(end)

                start_embedding = start_data.unsqueeze(0)
                end_embedding = end_data.unsqueeze(0)

                pred_direction_thetas = direction_network(start_embedding, end_embedding).squeeze(0)

                predicted_degree = radians_to_degrees(thetas_to_radians(pred_direction_thetas))
                expected_degree = direction_to_degrees_atan(direction)

                if math.fabs(predicted_degree - expected_degree) < 22.5:
                    win += 1
                else:
                    lose += 1

    print("")
    print("Win", win)
    print("Lose", lose)
    print("Win rate", win / (win + lose))


def run_tests_SSDir(direction_network, storage: StorageSuperset2):
    direction_network = direction_network.to(get_device())
    direction_network.eval()

    datapoints: List[str] = storage.get_all_datapoints()

    win = 0
    lose = 0
    ITERATIONS = 1

    print("")
    set_pretty_display(ITERATIONS * len(datapoints), "Iterations")
    pretty_display_start()

    for iter in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()
        for idx, datapoint in enumerate(datapoints):
            connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
            pretty_display(idx * (iter + 1))

            for j in range(len(connections_to_point)):
                start = connections_to_point[j]["start"]
                end = connections_to_point[j]["end"]
                direction = connections_to_point[j]["direction"]

                direction = torch.tensor(direction, dtype=torch.float32, device=get_device())
                l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
                direction = direction / l2_direction

                start_data = storage.get_datapoint_data_tensor_by_name_permuted(start).to(get_device()).unsqueeze(0)
                end_data = storage.get_datapoint_data_tensor_by_name_permuted(end).to(get_device()).unsqueeze(0)

                pred_direction_thetas = direction_network(start_data, end_data).squeeze(0)

                predicted_degree = radians_to_degrees(thetas_to_radians(pred_direction_thetas))
                expected_degree = direction_to_degrees_atan(direction)

                if math.fabs(predicted_degree - expected_degree) < 22.5:
                    win += 1
                else:
                    lose += 1

    print("")
    print("Win", win)
    print("Lose", lose)
    print("Win rate", win / (win + lose))
