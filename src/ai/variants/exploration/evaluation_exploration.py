import numpy as np
import torch
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.seen_network import SeenNetwork
from src.utils import array_to_tensor, get_device
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from typing import List
from src.modules.time_profiler import start_profiler, profiler_checkpoint, profiler_checkpoint_blank
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display
from src.ai.variants.exploration.utils import check_min_distance


def _get_connection_distances_raw_data_north(storage: StorageSuperset2) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(200, len(connections))

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
        # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

        start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
        end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)

        start_data_arr.append(start_data)
        end_data_arr.append(end_data)

    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())

    start_embedding = start_data_arr
    end_embedding = end_data_arr

    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distance_embeddings = distance_data

    length = len(distance_embeddings)

    for i in range(length):
        start_name = sampled_connections[i]["start"]
        end_name = sampled_connections[i]["end"]
        distance_real = sampled_connections[i]["distance"]
        distance_data_i = distance_data[i]
        distance_embeddings_i = distance_embeddings[i]
        connections_distances_data.append({
            "start": start_name,
            "end": end_name,
            "distance_real": distance_real,
            "distance_data": distance_data_i,
            "distance_embeddings": distance_embeddings_i
        })

    return connections_distances_data


def _get_connection_distances_seen_network(storage: StorageSuperset2, seen_network: SeenNetwork) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(100, len(connections))
    seen_network.eval()
    seen_network = seen_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
        end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)

        start_data_arr.append(start_data)
        end_data_arr.append(end_data)

    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())

    start_embedding = seen_network.encoder_inference(start_data_arr)
    end_embedding = seen_network.encoder_inference(end_data_arr)

    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distance_embeddings = torch.norm(start_embedding - end_embedding, p=2, dim=1)

    length = len(distance_embeddings)

    for i in range(length):
        start_name = sampled_connections[i]["start"]
        end_name = sampled_connections[i]["end"]
        distance_real = sampled_connections[i]["distance"]
        distance_data_i = distance_data[i].item()
        distance_embeddings_i = distance_embeddings[i].item()
        connections_distances_data.append({
            "start": start_name,
            "end": end_name,
            "distance_real": distance_real,
            "distance_data": distance_data_i,
            "distance_embeddings": distance_embeddings_i
        })

    return connections_distances_data


def print_distances_embeddings_inputs(storage: StorageSuperset2, seen_network: SeenNetwork):
    """
    Evaluates the relationship between the distances between the embeddings and the inputs
    """

    connections_distances_data = _get_connection_distances_seen_network(storage, seen_network)
    # sort by real distance
    connections_distances_data.sort(key=lambda x: x["distance_real"])
    for connection in connections_distances_data:
        print(f"{connection['distance_real']} => {connection['distance_embeddings']} || {connection['distance_data']}")


def eval_distances_threshold_averages_raw_data(storage: StorageSuperset2,
                                               real_distance_threshold):
    connections_distances_data = _get_connection_distances_raw_data_north(storage)

    REAL_DISTANCE_THRESHOLD = real_distance_threshold
    average_distance_embeddings = 0
    average_distance_data = 0
    total_count = 0

    for connection in connections_distances_data:
        real_distance = connection["distance_real"]
        if real_distance < REAL_DISTANCE_THRESHOLD:
            total_count += 1
            average_distance_embeddings += connection["distance_embeddings"]
            average_distance_data += connection["distance_data"]

    if total_count == 0:
        print("No connections found")
        return 0, 0
    average_distance_embeddings /= total_count
    average_distance_data /= total_count

    print(f"Average distance embeddings: {average_distance_embeddings}")
    print(f"Average distance data: {average_distance_data}")
    return average_distance_embeddings, average_distance_data


def eval_distances_threshold_averages_seen_network(storage: StorageSuperset2, seen_network: SeenNetwork,
                                                   real_distance_threshold):
    connections_distances_data = _get_connection_distances_seen_network(storage, seen_network)

    REAL_DISTANCE_THRESHOLD = real_distance_threshold
    average_distance_embeddings = 0
    average_distance_data = 0
    total_count = 0

    print("Connections distances data: ", connections_distances_data)

    for connection in connections_distances_data:
        real_distance = connection["distance_real"]
        if real_distance < REAL_DISTANCE_THRESHOLD:
            total_count += 1
            average_distance_embeddings += connection["distance_embeddings"]
            average_distance_data += connection["distance_data"]

    if total_count == 0:
        print("No connections found")
        return 0, 0
    average_distance_embeddings /= total_count
    average_distance_data /= total_count

    print(f"Average distance embeddings: {average_distance_embeddings}")
    print(f"Average distance data: {average_distance_data}")
    return average_distance_embeddings, average_distance_data
