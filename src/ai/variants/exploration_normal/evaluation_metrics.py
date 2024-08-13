import numpy as np
import torch
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration_normal.seen_network import SeenNetwork
from src.utils import array_to_tensor, get_device
from src.modules.time_profiler import start_profiler, profiler_checkpoint
from typing import List
from src.modules.time_profiler import start_profiler, profiler_checkpoint, profiler_checkpoint_blank
from src.modules.pretty_display import pretty_display_reset, pretty_display_start, pretty_display, set_pretty_display


def _get_connection_distances(storage: StorageSuperset2, seen_network: SeenNetwork) -> any:
    connections = storage.get_connections_data()
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

        start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
        end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)
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


def print_distances_embeddings_inputs(storage: StorageSuperset2, seen_network: SeenNetwork):
    """
    Evaluates the relationship between the distances between the embeddings and the inputs
    """

    connections_distances_data = _get_connection_distances(storage, seen_network)
    # sort by real distance
    connections_distances_data.sort(key=lambda x: x["distance_real"])
    for connection in connections_distances_data:
        print(f"{connection['distance_real']} => {connection['distance_embeddings']} || {connection['distance_data']}")


def eval_distances_threshold_averages(storage: StorageSuperset2, seen_network: SeenNetwork,
                                      real_distance_threshold):
    connections_distances_data = _get_connection_distances(storage, seen_network)

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


def _check_min_distance(storage: StorageSuperset2, datapoint):
    datapoints_names = storage.get_all_datapoints()
    current_x = datapoint["params"]["x"]
    current_y = datapoint["params"]["y"]
    current_name = datapoint["name"]
    adjcent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 2)
    adjcent_names.append(current_name)

    minimum_real_distance = 1000000

    for name in datapoints_names:
        if name in adjcent_names or name == current_name:
            continue

        data = storage.get_datapoint_by_name(name)

        data_x = data["params"]["x"]
        data_y = data["params"]["y"]
        data_name = name

        real_distance = np.sqrt((current_x - data_x) ** 2 + (current_y - data_y) ** 2)
        if real_distance < minimum_real_distance:
            minimum_real_distance = real_distance

    return minimum_real_distance


def evaluate_distance_metric(storage: StorageSuperset2, metric, new_datapoints: List[any],
                             distance_threshold):
    """
    Evaluate new datapoints and old datapoints with the distance metric
    """

    THRESHOLD = distance_threshold
    should_be_found = []
    should_not_be_found = []

    print("Started evaluating metric")
    # finds out what new datapoints should be found as adjacent
    for new_datapoint in new_datapoints:
        minimum_distance = _check_min_distance(storage, new_datapoint)
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
        distance = _check_min_distance(storage, false_positive)
        print("false positive", distance)
