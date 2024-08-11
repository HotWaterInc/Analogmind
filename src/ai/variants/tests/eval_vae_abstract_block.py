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


def evaluate_distances_between_pairs_vae_abstract(model: BaseAutoencoderModel, storage: StorageSuperset2,
                                                  rotations0: bool = False) -> float:
    """
    Gives the average distance between connected pairs ( degree 1 ) and non-connected pairs ( degree 2, 3, 4, etc. )
    """
    adjacent_data = storage.get_adjacency_data()
    non_adjacent_data = storage.get_all_adjacent_data()
    SAMPLED_NON_ADJCENT = min(5000, len(non_adjacent_data))
    non_adjacent_data = np.random.choice(np.array(non_adjacent_data), SAMPLED_NON_ADJCENT, replace=False)

    average_average_deg1_distance = 0
    average_avg_distances = {}
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # storage.build_permuted_data_raw()
    ITERATIONS = 5
    if rotations0:
        ITERATIONS = 1

    for iteration in range(ITERATIONS):
        # selects a rotation for each datapoint at each iteration
        if rotations0:
            storage.build_permuted_data_random_rotations_rotation0()
        else:
            storage.build_permuted_data_random_rotations()

        average_deg1_distance = 0
        avg_distances = {}
        max_distance = 0

        start_data_tensors = []
        end_data_tensors = []

        for connection in adjacent_data:
            start_uid = connection["start"]
            end_uid = connection["end"]
            distance = connection["distance"]

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start_uid)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end_uid)
            start_data_tensors.append(start_data)
            end_data_tensors.append(end_data)

        start_data_tensors = torch.stack(start_data_tensors).to(device)
        end_data_tensors = torch.stack(end_data_tensors).to(device)
        model = model.to(device)

        start_embedding, _ = model.encoder_inference(start_data_tensors)
        end_embedding, _ = model.encoder_inference(end_data_tensors)
        embedding_size = start_embedding.shape[1]
        distance_between_embeddings = torch.sum(torch.norm((start_embedding - end_embedding),
                                                           p=2, dim=1) / embedding_size).item()
        average_deg1_distance += distance_between_embeddings

        start_data_tensors = []
        end_data_tensors = []
        distances = []

        for connection in non_adjacent_data:
            start_uid = connection["start"]
            end_uid = connection["end"]
            distance = connection["distance"]
            distances.append(distance)
            max_distance = max(max_distance, distance)

            start_data = storage.get_datapoint_data_tensor_by_name_permuted(start_uid)
            end_data = storage.get_datapoint_data_tensor_by_name_permuted(end_uid)
            start_data_tensors.append(start_data)
            end_data_tensors.append(end_data)

        start_data_tensors = torch.stack(start_data_tensors).to(device)
        end_data_tensors = torch.stack(end_data_tensors).to(device)

        start_embedding, _ = model.encoder_inference(start_data_tensors)
        end_embedding, _ = model.encoder_inference(end_data_tensors)

        embedding_size = start_embedding.shape[1]
        distance_between_embeddings = torch.norm((start_embedding - end_embedding),
                                                 p=2, dim=1) / embedding_size
        for index, distance in enumerate(distances):
            if f"{distance}" not in avg_distances:
                avg_distances[f"{distance}"] = {
                    "sum": 0,
                    "count": 0
                }
            avg_distances[f"{distance}"]["sum"] += distance_between_embeddings[index].item()
            avg_distances[f"{distance}"]["count"] += 1

        average_deg1_distance /= len(adjacent_data)
        average_average_deg1_distance += average_deg1_distance

        for distance in range(2, max_distance + 1):
            avg_distances[f"{distance}"]["sum"] /= avg_distances[f"{distance}"]["count"]
            if f"{distance}" not in average_avg_distances:
                average_avg_distances[f"{distance}"] = {
                    "sum": 0,
                    "count": 0
                }
            average_avg_distances[f"{distance}"]["sum"] += avg_distances[f"{distance}"]["sum"]
            average_avg_distances[f"{distance}"]["count"] += 1

    average_average_deg1_distance /= ITERATIONS
    print(f"Average average distance between connected pairs: {average_average_deg1_distance:.4f}")

    for distance in range(2, max_distance + 1):
        if f"{distance}" in average_avg_distances:
            average_avg_distances[f"{distance}"]["sum"] /= average_avg_distances[f"{distance}"]["count"]
            print(
                f"Average average distance for distance {distance}: {average_avg_distances[f'{distance}']['sum']:.4f}")

    return average_average_deg1_distance


def evaluate_adjacency_properties_vae_abstract(model: BaseAutoencoderModel, storage: StorageSuperset2,
                                               average_distance_adjacent: float, rotation0: bool = False):
    """
    Evaluates how well the encoder finds adjacency in the data
    """
    embedding_size = model.get_embedding_size()
    distance_threshold = average_distance_adjacent * embedding_size * 1.25

    adjacent_data = storage.get_adjacency_data()
    non_adjacent_data = storage.get_non_adjacent_data()
    all_data = adjacent_data + non_adjacent_data

    total_pairs = len(all_data)
    true_adjacent_pairs = len(adjacent_data)
    true_non_adjacent_pairs = len(non_adjacent_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    average_false_positives = 0
    average_really_bad_false_positives = 0
    average_true_positives = 0
    average_true_adjacent_pairs = 0

    ITERATIONS = 3
    if rotation0:
        ITERATIONS = 1

    for iteration in range(ITERATIONS):
        if rotation0:
            storage.build_permuted_data_random_rotations_rotation0()
        else:
            storage.build_permuted_data_random_rotations()

        found_adjacent_pairs = []
        false_positives = []
        true_positives = []
        really_bad_false_positives = []

        start_tensors = []
        end_tensors = []
        real_life_distances = []
        start_uids = []
        end_uids = []

        for connection in all_data:
            start = connection["start"]
            end = connection["end"]
            real_life_distance = connection["distance"]

            start_tensors.append(storage.get_datapoint_data_tensor_by_name_permuted(start))
            end_tensors.append(storage.get_datapoint_data_tensor_by_name_permuted(end))
            real_life_distances.append(real_life_distance)
            start_uids.append(start)
            end_uids.append(end)

        start_tensors = torch.stack(start_tensors).to(device)
        end_tensors = torch.stack(end_tensors).to(device)
        model = model.to(device)
        model.eval()

        i_encoded, _ = model.encoder_inference(start_tensors)
        j_encoded, _ = model.encoder_inference(end_tensors)

        distances = torch.norm((i_encoded - j_encoded), p=2, dim=1)
        distances_items = distances.detach().cpu().numpy()

        for index, distance in enumerate(distances_items):
            real_life_distance = real_life_distances[index]
            start = start_uids[index]
            end = end_uids[index]
            if distance < distance_threshold:
                found_adjacent_pairs.append((start, end))
                if real_life_distance <= 1:
                    true_positives.append((start, end))

                if real_life_distance >= 2:
                    false_positives.append((start, end))

                if real_life_distance >= 3:
                    really_bad_false_positives.append((start, end))

        if len(found_adjacent_pairs) == 0:
            print("No adjacent pairs found in this iteration")
            return

        average_false_positives += len(false_positives) / len(found_adjacent_pairs) * 100
        average_really_bad_false_positives += len(really_bad_false_positives) / len(found_adjacent_pairs) * 100
        average_true_positives += len(true_positives) / len(found_adjacent_pairs) * 100
        average_true_adjacent_pairs += len(true_positives) / true_adjacent_pairs * 100

    average_false_positives /= ITERATIONS
    average_really_bad_false_positives /= ITERATIONS
    average_true_positives /= ITERATIONS
    average_true_adjacent_pairs /= ITERATIONS

    print("METRICS ADJACENCY --------------------------")
    print(f"Average percentage of false positives: {average_false_positives:.2f}%")
    print(f"Average percentage of DISTANT false positives: {average_really_bad_false_positives:.2f}%")
    print(f"Average percentage of true positives: {average_true_positives:.2f}%")
    print(f"Average percentage of adjacent pairs from total found: {average_true_adjacent_pairs:.2f}%")


def evaluate_confidence_vae_abstract(model: BaseAutoencoderModel, storage: StorageSuperset2,
                                     rotations0: bool = False) -> None:
    """
    Evaluates the reconstruction error on random samples from the training data
    """
    print("\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_averaged_var = 0
    total_averaged_logvar = 0
    model.eval()
    model.to(device)
    ITERATIONS = 10
    if rotations0:
        ITERATIONS = 1

    # First pass to calculate the global mean logvar
    for iteration in range(ITERATIONS):
        storage.build_permuted_data_random_rotations_rotation_N(0)

        train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))
        data = train_data.to(device)  # Add batch dimension
        position_mean, position_var, thetas_mean, thetas_var = model.encoder_training(data)
        position_var_tensor: torch.Tensor = logvar_to_variance(position_var)
        position_logvar_tensor: torch.Tensor = position_var
        total_error = position_var_tensor.mean(dim=1).mean().item()
        total_error_logvar = position_logvar_tensor.mean(dim=1).mean().item()
        total_averaged_var += total_error
        total_averaged_logvar += total_error_logvar

    final_avg_logvar = total_averaged_logvar / ITERATIONS

    # Second pass to calculate distances from mean and standard deviation
    all_logvars = []
    for iteration in range(ITERATIONS):
        if rotations0:
            storage.build_permuted_data_random_rotations_rotation_N(0)
        else:
            storage.build_permuted_data_random_rotations()

        train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))
        data = train_data.to(device)  # Add batch dimension
        position_mean, position_var, thetas_mean, thetas_var = model.encoder_training(data)
        position_logvar_tensor: torch.Tensor = position_var
        all_logvars.append(position_logvar_tensor.mean(dim=1).cpu().detach().numpy())

    all_logvars = np.concatenate(all_logvars)
    global_std = np.std(all_logvars)

    # Calculate distances and count datapoints outside 1, 2, and 3 standard deviations
    distances = np.abs(all_logvars - final_avg_logvar)
    outside_1sd = np.sum(distances > global_std)
    outside_2sd = np.sum(distances > 2 * global_std)
    outside_3sd = np.sum(distances > 3 * global_std)

    total_datapoints = len(all_logvars)

    print(f'Total average variation over {ITERATIONS} iterations: {total_averaged_var / ITERATIONS:.6f}')
    print(f'Total average logvar over {ITERATIONS} iterations: {total_averaged_logvar / ITERATIONS:.6f}')
    print(f'Global standard deviation of logvars: {global_std:.6f}')

    print(f'Datapoints outside 1 standard deviation: {outside_1sd} ({outside_1sd / total_datapoints * 100:.2f}%)')
    print(f'Datapoints outside 2 standard deviations: {outside_2sd} ({outside_2sd / total_datapoints * 100:.2f}%)')
    print(f'Datapoints outside 3 standard deviations: {outside_3sd} ({outside_3sd / total_datapoints * 100:.2f}%)')


def evaluate_reconstruct_vae_abstract(model: BaseAutoencoderModel, storage: StorageSuperset2,
                                      rotations0: bool = False) -> None:
    """
    Evaluates the reconstruction error on random samples from the training data
    """
    print("\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_averaged_error = 0
    model.eval()
    model.to(device)
    ITERATIONS = 10
    if rotations0:
        ITERATIONS = 1

    # First pass to calculate the global mean logvar
    for iteration in range(ITERATIONS):
        if rotations0:
            storage.build_permuted_data_random_rotations_rotation_N(0)
        else:
            storage.build_permuted_data_random_rotations()

        train_data = array_to_tensor(np.array(storage.get_pure_permuted_raw_env_data()))
        data = train_data.to(device)  # Add batch dimension
        decoded = model.forward_inference(data)

        total_error = torch.norm((data - decoded), p=2, dim=1).mean().item()
        total_averaged_error += total_error

    total_averaged_error /= ITERATIONS
    print(f'Total average reconstruction error over {ITERATIONS} iterations: {total_averaged_error:.6f}')


def logvar_to_variance(logvar: torch.Tensor) -> torch.Tensor:
    return torch.exp(logvar)
