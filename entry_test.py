from typing import List, Dict
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy.stats import pearsonr
import numpy as np
from fontTools.misc.cython import returns
from pyglet.input.linux.evdev import get_devices
import torch

from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.data_augmentation import load_storage_with_base_data
from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.params import MAX_DISTANCE
from src.ai.variants.exploration.utils_pure_functions import distance_thetas_to_distance_percent, flag_data_authenticity
from src.modules.pretty_display import pretty_display_start, pretty_display_set, pretty_display, pretty_display_reset
from src.modules.save_load_handlers.ai_models_handle import load_ai_version, load_other_ai, load_manually_saved_ai, \
    load_custom_ai
from src.modules.save_load_handlers.data_handle import read_other_data_from_file
from src.utils import get_device


def eval_data_changes(storage: StorageSuperset2, seen_network: any) -> any:
    connections = storage.get_all_connections_only_datapoints_authenticity_filter(authentic_direction=True,
                                                                                  authentic_distance=True)
    SAMPLES = min(200, len(connections))
    seen_network.eval()
    seen_network = seen_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    same_position_difference = 0
    different_position_difference = 0

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        start_rotations_arr = []
        end_rotations_arr = []
        for i in range(24):
            start_rotations_arr.append(storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, i))
            end_rotations_arr.append(storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, i))

        start_embeddings = seen_network.encoder_inference(torch.stack(start_rotations_arr).to(get_device()))
        end_embeddings = seen_network.encoder_inference(torch.stack(end_rotations_arr).to(get_device()))

        same_position_difference += torch.norm(start_embeddings[0] - start_embeddings[12], p=2, dim=0).mean().item()
        raw_diff = torch.norm(start_embeddings[0] - end_embeddings[12], p=2, dim=0).item()

        different_position_difference += raw_diff

        if raw_diff > 1:
            different_position_difference -= raw_diff

    same_position_difference /= SAMPLES
    different_position_difference /= SAMPLES

    print(f"Same position difference: {same_position_difference}")
    print(f"Different position difference: {different_position_difference}")


def _get_connection_distances_seen_network(storage: StorageSuperset2, seen_network: any) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(1000, len(connections))
    seen_network.eval()
    seen_network = seen_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        # start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 17)
        # end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 17)

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


def _get_connection_distances_adjacency_network_on_unknown_dataset(storage: StorageSuperset2,
                                                                   adjacency_network: any) -> any:
    adjacency_network.eval()
    adjacency_network = adjacency_network.to(get_device())

    datapoints = storage.get_all_datapoints()

    distances_arr = []
    start_data_arr = []
    end_data_arr = []

    pretty_display_set(len(datapoints), "Calculating distances")
    pretty_display_start()

    lng = len(datapoints)
    start_names = []
    end_names = []

    for i in range(lng):
        pretty_display(i)
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            distance = storage.get_datapoints_real_distance(start_name, end_name)

            start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
            end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)
            # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
            # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

            start_names.append(start_name)
            end_names.append(end_name)
            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            distances_arr.append(distance)

    print("")
    print("finished first loop")
    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())
    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    adjacency_probabilities = adjacency_network(start_data_arr, end_data_arr)
    print("finished forwarding")

    pretty_display_set(len(adjacency_probabilities), "Calculating distances from thetas")
    pretty_display_start()

    predicted_adjacencies = []
    for idx, distance in enumerate(adjacency_probabilities):
        if distance[0] > 0.98:
            predicted_adjacencies.append(0)
        else:
            predicted_adjacencies.append(1)

    print("")
    index = 0

    good_predictions = 0
    bad_predictions = 0
    bad_pred_avg_distance = 0

    expected_good_predictions = 0

    neigh_net = load_manually_saved_ai("neigh_network_north.pth")
    neigh_net.eval()
    neigh_net = neigh_net.to(get_device())

    predicted_distances = []
    expected_distances = []

    for i in range(lng):
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            # print(start_names[index], end_names[index])
            distance_real = storage.get_datapoints_real_distance(start_name, end_name)

            predicted_adjacency = predicted_adjacencies[index]
            # 0 true, 1 false
            if predicted_adjacency == 0:
                if distance_real < 0.5:
                    good_predictions += 1


                elif distance_real > 1.25:
                    bad_predictions += 1
                    bad_pred_avg_distance += distance_real

                start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0).to(
                    get_device())
                end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0).to(get_device())
                distance_thetas = neigh_net(start_data.unsqueeze(0), end_data.unsqueeze(0)).squeeze(0)
                distance_percent = distance_thetas_to_distance_percent(distance_thetas)
                distance_percent *= MAX_DISTANCE

                predicted_distances.append(distance_percent)
                expected_distances.append(distance_real)
                if distance_real > 0.75:
                    print(f"Predicted distance: {distance_percent}, real distance: {distance_real}")

            if distance_real < 0.5:
                expected_good_predictions += 1

            index += 1

    se = 0
    for i in range(len(expected_distances)):
        se += (expected_distances[i] - predicted_distances[i]) ** 2

    se /= len(expected_distances)
    se = se ** 0.5
    print(f"Standard error: {se}")

    if bad_predictions > 0:
        bad_pred_avg_distance /= bad_predictions
        print(f"Bad predictions avg distance: {bad_pred_avg_distance}")

    print(f"Expected good predictions: {expected_good_predictions}")
    print(f"Good predictions: {good_predictions}")
    print(f"Bad predictions: {bad_predictions}")

    print("finished second loop")


def standard_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")

    # Compute squared differences
    squared_diff = np.square(np.array(y_true) - np.array(y_pred))

    # Calculate the standard error
    se = np.sqrt(np.mean(squared_diff))

    return se


def _get_mse_losses_seen_network_on_dataset(storage: StorageSuperset2,
                                            network: BaseAutoencoderModel) -> any:
    network = network.to(get_device())
    network.eval()

    datapoints = storage.get_all_datapoints()

    axis_coord = []
    data_arr = []
    mse_losses = []

    pretty_display_set(len(datapoints), "Calculating distances")
    pretty_display_start()

    lng = len(datapoints)

    for i in range(lng):
        pretty_display(i)
        name = datapoints[i]
        metadata = storage.get_datapoint_metadata_coords(name)

        data = storage.get_datapoint_data_random_rotation_tensor_by_name(name)
        coord = metadata[1]

        axis_coord.append(coord)
        data_arr.append(data)

    data_arr = torch.stack(data_arr).to(get_device())
    data_reconstruction = network.forward_inference(data_arr).to(get_device())
    mse_loss = torch.norm(data_reconstruction - data_arr, p=1, dim=1)

    datapoints_reconstructions = []
    for i in range(lng):
        name = datapoints[i]
        axis = axis_coord[i]
        mse_loss_i = mse_loss[i].item()

        datapoints_reconstructions.append({
            "start": name,
            "coord": axis,
            "mse_loss": mse_loss_i
        })

    return datapoints_reconstructions


def _get_connection_distances_on_some_network_on_unknown_dataset(storage: StorageSuperset2,
                                                                 network: BaseAutoencoderModel) -> any:
    network.eval()
    network = network.to(get_device())

    datapoints = storage.get_all_datapoints()

    distances_arr = []
    start_data_arr = []
    end_data_arr = []

    pretty_display_set(len(datapoints), "Calculating distances")
    pretty_display_start()

    lng = len(datapoints)
    upper_i = 100

    for i in range(lng):
        pretty_display(i)
        for j in range(i + 1, min(i + upper_i, lng)):
            start_name = datapoints[i]
            end_name = datapoints[j]
            distance = storage.get_datapoints_walking_distance(start_name, end_name)

            start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
            end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            distances_arr.append(distance)

    print("")
    print("finished first loop")
    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())
    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)

    start_embeddings = network.encoder_inference(start_data_arr).to(get_device())
    end_embeddings = network.encoder_inference(end_data_arr).to(get_device())

    raw_diff_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    raw_diff_embeddings = torch.norm(start_embeddings - end_embeddings, p=2, dim=1)

    final_connections = []
    cnt = 0
    for i in range(lng):
        for j in range(i + 1, min(i + upper_i, lng)):
            start_name = datapoints[i]
            end_name = datapoints[j]

            distance_real = storage.get_datapoints_real_distance(start_name, end_name)
            distance_data = raw_diff_data[cnt].item()
            distance_embeddings = raw_diff_embeddings[cnt].item()

            # if distance_real > 1:
            #     # cnt += 1
            #     continue

            final_connections.append({
                "start": start_name,
                "end": end_name,
                "distance_real": distance_real,
                "distance_data": distance_data,
                "distance_embeddings": distance_embeddings
            })
            cnt += 1

    print("finished second loop")

    return final_connections


def _get_connection_distances_neigh_network_on_unknown_dataset(storage: StorageSuperset2,
                                                               neighborhood_network: any) -> any:
    neighborhood_network.eval()
    neighborhood_network = neighborhood_network.to(get_device())

    datapoints = storage.get_all_datapoints()

    distances_arr = []
    start_data_arr = []
    end_data_arr = []

    pretty_display_set(len(datapoints), "Calculating distances")
    pretty_display_start()

    lng = len(datapoints)
    for i in range(lng):
        pretty_display(i)
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            distance = storage.get_datapoints_real_distance(start_name, end_name)
            if distance > 1:
                continue

            start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
            end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)
            # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
            # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            distances_arr.append(distance)

    print("")
    print("finished first loop")
    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())
    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distances_thetas = neighborhood_network(start_data_arr, end_data_arr)
    print("finished forwarding")

    pretty_display_set(len(distances_thetas), "Calculating distances from thetas")
    pretty_display_start()

    predicted_distances = []
    print(len(distances_thetas))
    for idx, distance in enumerate(distances_thetas):
        pretty_display(idx)
        distance_percent = distance_thetas_to_distance_percent(distance)
        distance_percent *= MAX_DISTANCE
        predicted_distances.append(distance_percent)

    print("")
    print("finished making distajcces")

    final_connections = []
    cnt = 0
    for i in range(lng):
        for j in range(i + 1, lng):
            start_name = datapoints[i]
            end_name = datapoints[j]
            distance_real = storage.get_datapoints_real_distance(start_name, end_name)

            if distance_real > 1:
                # cnt += 1
                continue

            distance_data_i = distance_data[cnt].item()
            predicted_distance = predicted_distances[cnt].item()
            # predicted_distance = 0

            final_connections.append({
                "start": start_name,
                "end": end_name,
                "distance_real": distance_real,
                "distance_data": distance_data_i,
                "distance_embeddings": predicted_distance
            })
            cnt += 1

    print("finished second loop")

    return final_connections


def _get_connection_distances_neigh_network_on_training_dataset(storage: StorageSuperset2,
                                                                neighborhood_network: any) -> any:
    connections = storage.get_all_connections_data()
    SAMPLES = min(1000, len(connections))
    neighborhood_network.eval()
    neighborhood_network = neighborhood_network.to(get_device())

    sampled_connections = np.random.choice(np.array(connections), SAMPLES, replace=False)
    connections_distances_data = []

    start_data_arr = []
    end_data_arr = []

    for connection in sampled_connections:
        start_name = connection["start"]
        end_name = connection["end"]

        distance = storage.get_datapoints_real_distance(start_name, end_name)
        start_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(start_name, 0)
        end_data = storage.get_datapoint_data_selected_rotation_tensor_by_name(end_name, 0)

        # start_data = storage.get_datapoint_data_random_rotation_tensor_by_name(start_name)
        # end_data = storage.get_datapoint_data_random_rotation_tensor_by_name(end_name)

        start_data_arr.append(start_data)
        end_data_arr.append(end_data)

    start_data_arr = torch.stack(start_data_arr).to(get_device())
    end_data_arr = torch.stack(end_data_arr).to(get_device())
    distance_data = torch.norm(start_data_arr - end_data_arr, p=2, dim=1)
    distances_thetas = neighborhood_network(start_data_arr, end_data_arr)

    predicted_distances = []
    for distance in distances_thetas:
        distance_percent = distance_thetas_to_distance_percent(distance)
        distance_percent *= MAX_DISTANCE
        predicted_distances.append(distance_percent)

    for i in range(SAMPLES):
        start_name = sampled_connections[i]["start"]
        end_name = sampled_connections[i]["end"]
        distance_real = sampled_connections[i]["distance"]

        distance_data_i = distance_data[i].item()
        predicted_distance = predicted_distances[i].item()
        # predicted_distance = 0

        connections_distances_data.append({
            "start": start_name,
            "end": end_name,
            "distance_real": distance_real,
            "distance_data": distance_data_i,
            "distance_embeddings": predicted_distance
        })

    return connections_distances_data


def calculate_pearson_correlations(data: ndarray):
    data_array = np.array(data)

    # Extract columns
    input_values = data_array[:, 0]
    second_number = data_array[:, 1]
    third_number = data_array[:, 2]

    # Pearson correlation
    pearson_input_second = pearsonr(input_values, second_number)
    pearson_input_third = pearsonr(input_values, third_number)
    pearson_second_third = pearsonr(second_number, third_number)

    # Print correlation results
    print("Pearson Correlation Coefficients:")
    print(f"Input vs Second Number: {pearson_input_second[0]:.3f} (p-value: {pearson_input_second[1]:.3e})")
    print(f"Input vs Third Number: {pearson_input_third[0]:.3f} (p-value: {pearson_input_third[1]:.3e})")
    print(f"Second Number vs Third Number: {pearson_second_third[0]:.3f} (p-value: {pearson_second_third[1]:.3e})")


def get_manifold_datapoint_distances():
    random_walk_datapoints = read_other_data_from_file(f"(1)_datapoints_autonomous_walk.json")
    random_walk_connections = read_other_data_from_file(f"(1)_connections_autonomous_walk_augmented_filled.json")
    storage: StorageSuperset2 = StorageSuperset2()
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    abs_network = load_custom_ai(model_name="manifold_network_normal", folder_name="manually_saved")
    connections = _get_connection_distances_on_some_network_on_unknown_dataset(storage, abs_network)
    return connections


def get_reconstructions_seen_network():
    random_walk_datapoints = read_other_data_from_file(f"(1)_datapoints_autonomous_walk.json")
    random_walk_connections = read_other_data_from_file(f"(1)_connections_autonomous_walk_augmented_filled.json")
    random_walk_connections = flag_data_authenticity(random_walk_connections)
    storage: StorageSuperset2 = StorageSuperset2()
    storage.incorporate_new_data(random_walk_datapoints, random_walk_connections)

    seen_network = load_custom_ai(model_name="seen_network_01.pth", folder_name="exploration_inference_no_sofa")
    datapoints = _get_mse_losses_seen_network_on_dataset(storage, seen_network)
    return datapoints


def plot_datapoints(subplot: int, x_axis_data: List[any], y_axis_data: List[any], x_label: str, y_label: str,
                    title: str, color: str, alpha: float = 0.7):
    plt.subplot(subplot)
    plt.scatter(x_axis_data, y_axis_data, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def visualize_datapoints_reconstructions(datapoints_recons):
    filtered_datapoints = [
        connection for connection in datapoints_recons
    ]
    data_arr = [[connection["coord"], connection["mse_loss"]] for
                connection in filtered_datapoints]
    data_array = np.array(data_arr)
    filtered_data = data_array

    # Create the plot
    plt.figure(figsize=(4, 4))

    plot_datapoints(131, filtered_data[:, 0], filtered_data[:, 1], "X axis coords", "Reconstruction error",
                    "Real vs Data Distance", "blue")

    plt.tight_layout()
    plt.show()


def visualize_connections_distances(connections):
    filtered_connections = [
        connection for connection in connections
        # if connection["distance_real"] < 2
    ]
    data_arr = [[connection["distance_real"], connection["distance_data"], connection["distance_embeddings"]] for
                connection in filtered_connections]
    data_array = np.array(data_arr)
    filtered_data = data_array

    # Create the plot
    plt.figure(figsize=(5, 4))

    # plot_datapoints(132, filtered_data[:, 0], filtered_data[:, 2], "Walking Distance", "S Distance",
    #                 "Walking distance vs S distances", "red")
    plt.scatter(filtered_data[:, 0], filtered_data[:, 2], color="red", alpha=0.7)
    plt.title("Walking distance vs S distances")
    plt.xlabel("Walking Distance")
    plt.ylabel("S Distance")

    plt.tight_layout()
    plt.show()
    calculate_pearson_correlations(filtered_data)


if __name__ == "__main__":
    connections = get_manifold_datapoint_distances()
    visualize_connections_distances(connections)

    # datapoints = get_reconstructions_seen_network()
    # visualize_datapoints_reconstructions(datapoints)
    pass
