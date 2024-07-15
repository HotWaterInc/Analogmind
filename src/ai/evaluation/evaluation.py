import numpy as np
import torch
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage import Storage
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.utils import array_to_tensor


def evaluate_reconstruction_error_super(model: BaseAutoencoderModel, storage: StorageSuperset2) -> None:
    """
    Evaluates the reconstruction error on random samples from the training data
    """
    print("\n")
    print("Evaluation on random samples from training data:")

    nr_of_samples = 64
    total_averaged_error = 0
    for iteration in range(10):
        storage.build_permuted_data_random_rotations()
        train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))
        indices = np.random.choice(len(train_data), nr_of_samples, replace=False)
        total_error = 0
        with torch.no_grad():
            for i, idx in enumerate(indices):
                data = train_data[idx].unsqueeze(0)  # Add batch dimension
                reconstructed = model.forward_inference(data)
                total_error += torch.sum(torch.abs(data - reconstructed)).item()

        sensor_datapoints = storage.metadata["sensor_distance"]
        # print(
        #     f'Iteration {iteration} average error per sample is {total_error / (nr_of_samples * sensor_datapoints):.4f}')
        total_averaged_error += total_error / (nr_of_samples * sensor_datapoints)

    print(f'Total average error over 10 iterations: {total_averaged_error / 10:.4f}')


def evaluate_distances_between_pairs_super(model: BaseAutoencoderModel, storage: StorageSuperset2) -> float:
    """
    Gives the average distance between connected pairs ( degree 1 ) and non-connected pairs ( degree 2, 3, 4, etc. )
    """
    adjacent_data = storage.get_adjacency_data()
    non_adjacent_data = storage.get_all_adjacent_data()

    average_average_deg1_distance = 0
    average_avg_distances = {}

    # storage.build_permuted_data_raw()

    for iteration in range(5):
        # selects a rotation for each datapoint at each iteration
        storage.build_permuted_data_random_rotations()
        average_deg1_distance = 0
        avg_distances = {}
        max_distance = 0

        for connection in adjacent_data:
            start_uid = connection["start"]
            end_uid = connection["end"]
            distance = connection["distance"]

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start_uid)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end_uid)

            start_embedding = model.encoder_inference(start_data.unsqueeze(0))
            end_embedding = model.encoder_inference(end_data.unsqueeze(0))

            distance_between_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()
            average_deg1_distance += distance_between_embeddings

        for connection in non_adjacent_data:
            start_uid = connection["start"]
            end_uid = connection["end"]
            distance = connection["distance"]
            max_distance = max(max_distance, distance)

            start_data = storage.get_datapoint_data_tensor_by_name(start_uid)
            end_data = storage.get_datapoint_data_tensor_by_name(end_uid)

            start_embedding = model.encoder_inference(start_data.unsqueeze(0))
            end_embedding = model.encoder_inference(end_data.unsqueeze(0))

            distance_between_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()

            if f"{distance}" not in avg_distances:
                avg_distances[f"{distance}"] = {
                    "sum": 0,
                    "count": 0
                }

            avg_distances[f"{distance}"]["sum"] += distance_between_embeddings
            avg_distances[f"{distance}"]["count"] += 1

        average_deg1_distance /= len(adjacent_data)
        average_average_deg1_distance += average_deg1_distance

        for distance in range(2, max_distance + 1):
            if f"{distance}" in avg_distances:
                avg_distances[f"{distance}"]["sum"] /= avg_distances[f"{distance}"]["count"]
                if f"{distance}" not in average_avg_distances:
                    average_avg_distances[f"{distance}"] = {
                        "sum": 0,
                        "count": 0
                    }
                average_avg_distances[f"{distance}"]["sum"] += avg_distances[f"{distance}"]["sum"]
                average_avg_distances[f"{distance}"]["count"] += 1

    average_average_deg1_distance /= 5
    print(f"Average average distance between connected pairs: {average_average_deg1_distance:.4f}")

    for distance in range(2, max_distance + 1):
        if f"{distance}" in average_avg_distances:
            average_avg_distances[f"{distance}"]["sum"] /= average_avg_distances[f"{distance}"]["count"]
            print(
                f"Average average distance for distance {distance}: {average_avg_distances[f'{distance}']['sum']:.4f}")

    return average_average_deg1_distance


def evaluate_adjacency_properties_super(model: BaseAutoencoderModel, storage: StorageSuperset2,
                                        average_distance_adjacent: float):
    """
    Evaluates how well the encoder finds adjacency in the data
    """
    distance_threshold = average_distance_adjacent * 1.25

    adjacent_data = storage.get_adjacency_data()
    non_adjacent_data = storage.get_non_adjacent_data()
    all_data = adjacent_data + non_adjacent_data

    total_pairs = len(all_data)
    true_adjacent_pairs = len(adjacent_data)
    true_non_adjacent_pairs = len(non_adjacent_data)

    # storage.build_permuted_data_raw()

    average_false_positives = 0
    average_really_bad_false_positives = 0
    average_true_positives = 0
    average_true_adjacent_pairs = 0

    ITERATIONS = 3
    for iteration in range(ITERATIONS):
        storage.build_permuted_data_random_rotations()

        found_adjacent_pairs = []
        false_positives = []
        true_positives = []
        really_bad_false_positives = []

        for connection in all_data:
            start = connection["start"]
            end = connection["end"]
            real_life_distance = connection["distance"]

            i_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name_permuted(start).unsqueeze(0))
            j_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name_permuted(end).unsqueeze(0))
            distance = torch.norm((i_encoded - j_encoded), p=2).item()

            if distance < distance_threshold:
                found_adjacent_pairs.append((start, end))
                if real_life_distance == 1:
                    true_positives.append((start, end))

                if real_life_distance > 1:
                    false_positives.append((start, end))

                if real_life_distance > 3:
                    really_bad_false_positives.append((start, end))

        # print("PREVIOUS METRICS --------------------------")
        # print(f"Number of FOUND adjacent pairs: {len(found_adjacent_pairs)}")
        # print(f"Number of FOUND adjacent false positives: {len(false_positives)}")
        # print(f"Number of FOUND adjacent DISTANT false positives: {len(really_bad_false_positives)}")
        # print(f"Number of FOUND TRUE adjacent pairs: {len(true_positives)}")
        # print(
        #     f"Total number of pairs: {total_pairs} made of {true_adjacent_pairs} adjacent and {true_non_adjacent_pairs} non-adjacent pairs.")
        #
        # if len(found_adjacent_pairs) == 0:
        #     return
        # print(f"Percentage of false positives: {len(false_positives) / len(found_adjacent_pairs) * 100:.2f}%")
        # print(
        #     f"Percentage of DISTANT false positives: {len(really_bad_false_positives) / len(found_adjacent_pairs) * 100:.2f}%")
        # print(f"Percentage of true positives: {len(true_positives) / len(found_adjacent_pairs) * 100:.2f}%")
        # print(f"Percentage of adjacent pairs from total found: {len(true_positives) / true_adjacent_pairs * 100:.2f}%")
        # print("--------------------------------------------------")

        average_false_positives += len(false_positives) / len(found_adjacent_pairs) * 100
        average_really_bad_false_positives += len(really_bad_false_positives) / len(found_adjacent_pairs) * 100
        average_true_positives += len(true_positives) / len(found_adjacent_pairs) * 100
        average_true_adjacent_pairs += len(true_positives) / true_adjacent_pairs * 100

        # # points which are far from each other yet have very close embeddings
        # sinking_points = 0
        # for connection in all_data:
        #     start = connection["start"]
        #     end = connection["end"]
        #     real_life_distance = connection["distance"]
        #
        #     i_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(start).unsqueeze(0))
        #     j_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(end).unsqueeze(0))
        #     distance = torch.norm((i_encoded - j_encoded), p=2).item()
        #
        #     if distance < distance_threshold and real_life_distance > 3:
        #         sinking_points += 1
        #
        # print("NEW METRICS --------------------------")
        # print(f"Number of sinking points: {sinking_points}")

    average_false_positives /= ITERATIONS
    average_really_bad_false_positives /= ITERATIONS
    average_true_positives /= ITERATIONS
    average_true_adjacent_pairs /= ITERATIONS

    print("METRICS ADJACENCY --------------------------")
    print(f"Average percentage of false positives: {average_false_positives:.2f}%")
    print(f"Average percentage of DISTANT false positives: {average_really_bad_false_positives:.2f}%")
    print(f"Average percentage of true positives: {average_true_positives:.2f}%")
    print(f"Average percentage of adjacent pairs from total found: {average_true_adjacent_pairs:.2f}%")


def evaluate_reconstruction_error(model: BaseAutoencoderModel, storage: Storage) -> None:
    """
    Evaluates the reconstruction error on random samples from the training data
    """
    print("\n")
    print("Evaluation on random samples from training data:")

    nr_of_samples = 64
    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data()))
    indices = np.random.choice(len(train_data), nr_of_samples, replace=False)
    total_error = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = train_data[idx].unsqueeze(0)  # Add batch dimension
            reconstructed = model.forward_inference(data)
            total_error += torch.sum(torch.abs(data - reconstructed)).item()

    sensor_datapoints = storage.metadata["sensor_distance"]
    print(
        f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error / (nr_of_samples * sensor_datapoints):.4f}')


def evaluate_distances_between_pairs(model: BaseAutoencoderModel, storage: Storage) -> float:
    """
    Gives the average distance between connected pairs ( degree 1 ) and non-connected pairs ( degree 2, 3, 4, etc. )
    """
    adjacent_data = storage.get_adjacency_data()
    non_adjacent_data = storage.get_all_adjacent_data()

    average_deg1_distance = 0
    avg_distances = {}
    max_distance = 0

    for connection in adjacent_data:
        start_uid = connection["start"]
        end_uid = connection["end"]
        distance = connection["distance"]

        start_data = storage.get_datapoint_data_tensor_by_name(start_uid)
        end_data = storage.get_datapoint_data_tensor_by_name(end_uid)

        start_embedding = model.encoder_inference(start_data.unsqueeze(0))
        end_embedding = model.encoder_inference(end_data.unsqueeze(0))

        distance_between_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()
        average_deg1_distance += distance_between_embeddings

    for connection in non_adjacent_data:
        start_uid = connection["start"]
        end_uid = connection["end"]
        distance = connection["distance"]
        max_distance = max(max_distance, distance)

        start_data = storage.get_datapoint_data_tensor_by_name(start_uid)
        end_data = storage.get_datapoint_data_tensor_by_name(end_uid)

        start_embedding = model.encoder_inference(start_data.unsqueeze(0))
        end_embedding = model.encoder_inference(end_data.unsqueeze(0))

        distance_between_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()

        if f"{distance}" not in avg_distances:
            avg_distances[f"{distance}"] = {
                "sum": 0,
                "count": 0
            }

        avg_distances[f"{distance}"]["sum"] += distance_between_embeddings
        avg_distances[f"{distance}"]["count"] += 1

    average_deg1_distance /= len(adjacent_data)
    print(f"Average distance between connected pairs: {average_deg1_distance:.4f}")

    for distance in range(2, max_distance + 1):
        if f"{distance}" in avg_distances:
            avg_distances[f"{distance}"]["sum"] /= avg_distances[f"{distance}"]["count"]
            print(f"Average distance for distance {distance}: {avg_distances[f'{distance}']['sum']:.4f}")

    return average_deg1_distance


def evaluate_adjacency_properties(model: BaseAutoencoderModel, storage: Storage, average_distance_adjacent: float):
    """
    Evaluates how well the encoder finds adjacency in the data
    """
    distance_threshold = average_distance_adjacent * 1.25

    adjacent_data = storage.get_adjacency_data()
    non_adjacent_data = storage.get_non_adjacent_data()
    all_data = adjacent_data + non_adjacent_data

    found_adjacent_pairs = []
    false_positives = []
    true_positives = []

    really_bad_false_positives = []

    total_pairs = len(all_data)
    true_adjacent_pairs = len(adjacent_data)
    true_non_adjacent_pairs = len(non_adjacent_data)

    for connection in all_data:
        start = connection["start"]
        end = connection["end"]
        real_life_distance = connection["distance"]

        i_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(start).unsqueeze(0))
        j_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(end).unsqueeze(0))
        distance = torch.norm((i_encoded - j_encoded), p=2).item()

        if distance < distance_threshold:
            found_adjacent_pairs.append((start, end))
            if real_life_distance == 1:
                true_positives.append((start, end))

            if real_life_distance > 1:
                false_positives.append((start, end))

            if real_life_distance > 2:
                really_bad_false_positives.append((start, end))

    print("PREVIOUS METRICS --------------------------")
    print(f"Number of FOUND adjacent pairs: {len(found_adjacent_pairs)}")
    print(f"Number of FOUND adjacent false positives: {len(false_positives)}")
    print(f"Number of FOUND adjacent DISTANT false positives: {len(really_bad_false_positives)}")
    print(f"Number of FOUND TRUE adjacent pairs: {len(true_positives)}")
    print(
        f"Total number of pairs: {total_pairs} made of {true_adjacent_pairs} adjacent and {true_non_adjacent_pairs} non-adjacent pairs.")

    if len(found_adjacent_pairs) == 0:
        return
    print(f"Percentage of false positives: {len(false_positives) / len(found_adjacent_pairs) * 100:.2f}%")
    print(
        f"Percentage of DISTANT false positives: {len(really_bad_false_positives) / len(found_adjacent_pairs) * 100:.2f}%")
    print(f"Percentage of true positives: {len(true_positives) / len(found_adjacent_pairs) * 100:.2f}%")
    print(f"Percentage of adjacent pairs from total found: {len(true_positives) / true_adjacent_pairs * 100:.2f}%")
    print("--------------------------------------------------")

    # points which are far from each other yet have very close embeddings
    sinking_points = 0
    for connection in all_data:
        start = connection["start"]
        end = connection["end"]
        real_life_distance = connection["distance"]

        i_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(start).unsqueeze(0))
        j_encoded = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(end).unsqueeze(0))
        distance = torch.norm((i_encoded - j_encoded), p=2).item()

        if distance < distance_threshold and real_life_distance > 3:
            sinking_points += 1

    print("NEW METRICS --------------------------")
    print(f"Number of sinking points: {sinking_points}")
