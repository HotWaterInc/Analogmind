import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2, RawConnectionData, normalize_direction, \
    degrees_to_percent, distance_percent_to_distance_thetas
from src.modules.save_load_handlers.ai_models_handle import load_manually_saved_ai, save_ai_manually
from src.utils import array_to_tensor, get_device
from typing import List
import torch.nn.functional as F
from src.modules.pretty_display import pretty_display, pretty_display_set, pretty_display_start, pretty_display_reset
from src.ai.runtime_data_storage.storage_superset2 import thetas_to_radians, \
    angle_percent_to_thetas_normalized_cached, \
    radians_to_degrees, atan2_to_standard_radians, radians_to_percent, coordinate_pair_to_radians_cursed_tranform, \
    direction_to_degrees_atan
from src.ai.variants.blocks import ResidualBlockSmallBatchNorm


def run_tests_SSDir_unseen(direction_network, storage: StorageSuperset2):
    direction_network = direction_network.to(get_device())
    direction_network.eval()

    datapoints: List[str] = storage.get_all_datapoints()

    win = 0
    lose = 0
    ITERATIONS = 1

    error_arr = []

    pretty_display_set(ITERATIONS * len(datapoints), "Iterations")
    pretty_display_start()

    for iter in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()
        for idx, datapoint in enumerate(datapoints):
            connections_to_point: List[
                RawConnectionData] = storage.get_datapoints_adjacent_at_degree_n_as_raw_connection_data(datapoint, 3)

            pretty_display(idx * (iter + 1))

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

    print("")
    pretty_display_set(len(datapoints), "Iterations")
    pretty_display_start()

    start_data_arr = []
    end_data_arr = []
    expected_thetas_arr = []

    storage.build_permuted_data_random_rotations()
    for idx, datapoint in enumerate(datapoints):
        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)
        pretty_display(idx)

        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]
            direction = connections_to_point[j]["direction"]

            direction = torch.tensor(direction, dtype=torch.float32, device=get_device())
            l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
            direction = direction / l2_direction

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start).to(get_device())
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end).to(get_device())

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            expected_thetas_arr.append(direction)

    print("Running network and evaluating ... ")
    start_embeddings_batch = torch.stack(start_data_arr).to(get_device())
    end_embeddings_batch = torch.stack(end_data_arr).to(get_device())

    pred_direction_thetas = direction_network(start_embeddings_batch, end_embeddings_batch)

    for idx, prediction in enumerate(pred_direction_thetas):
        expected_direction = expected_thetas_arr[idx]

        predicted_degree = radians_to_degrees(thetas_to_radians(prediction))
        expected_degree = direction_to_degrees_atan(expected_direction)

        if math.fabs(predicted_degree - expected_degree) < 22.5:
            win += 1
        else:
            lose += 1

    print("")
    print("Win", win)
    print("Lose", lose)
    print("Win rate", win / (win + lose))


def run_tests_SDirDistState(direction_network_SDDS, storage):
    direction_network_SDDS = direction_network_SDDS.to(get_device())
    direction_network_SDDS.eval()

    datapoints: List[str] = storage.get_all_datapoints()

    average_error = 0
    error_arr = []
    storage.build_permuted_data_random_rotations()

    start_data_arr = []
    end_data_arr = []
    direction_thetas_arr = []
    distance_thetas_arr = []

    print("Evaluating ... ")

    for datapoint in datapoints:
        connections_to_point: List[RawConnectionData] = storage.get_datapoint_adjacent_connections(datapoint)

        for j in range(len(connections_to_point)):
            start = connections_to_point[j]["start"]
            end = connections_to_point[j]["end"]
            direction = connections_to_point[j]["direction"]
            direction = normalize_direction(direction)
            direction_angle = direction_to_degrees_atan(direction)

            distance = connections_to_point[j]["distance"]
            distance_percent = distance / MAX_DISTANCE

            direction_thetas = angle_percent_to_thetas_normalized_cached(degrees_to_percent(direction_angle),
                                                                         DIRECTION_THETAS_SIZE)
            distance_thetas = distance_percent_to_distance_thetas(distance_percent, DISTANCE_THETAS_SIZE)
            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start).to(get_device())
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end).to(get_device())

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            direction_thetas_arr.append(direction_thetas)
            distance_thetas_arr.append(distance_thetas)

    start_embeddings_batch = torch.stack(start_data_arr).to(get_device())
    end_manifold_batch = torch.stack(end_data_arr).to(get_device())
    direction_thetas_batch = torch.stack(direction_thetas_arr).to(get_device())
    distance_thetas_batch = torch.stack(distance_thetas_arr).to(get_device())

    predicted_manifold = direction_network_SDDS(start_embeddings_batch, direction_thetas_batch, distance_thetas_batch)

    error_datapoint = torch.norm(predicted_manifold - end_manifold_batch, p=2, dim=1,
                                 keepdim=True).mean().item()

    print(f"Average error: {error_datapoint}")
