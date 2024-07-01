import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from src.modules.data_handlers.ai_models_handle import save_ai, load_latest_ai
from src.modules.data_handlers.parameters import *
from .parameters import DISTANCE_THRESHOLD
from src.ai.data_processing.ai_data_processing import preprocess_data
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.data_processing.ai_data_processing import normalize_data_min_max, normalize_data_z_score

storage: Storage = None


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 8),
            nn.Sigmoid()
        )

    def encoder(self, x):
        l1 = self.encoder1(x)
        encoded = self.encoder2(l1)
        deenc = self.encoder3(encoded)
        return encoded

    def forward(self, x):
        l1 = self.encoder1(x)
        encoded = self.encoder2(l1)
        deenc = self.encoder3(encoded)
        decoded = self.decoder(deenc)
        return encoded, decoded


model = Autoencoder()


def reconstruction_handling(autoencoder: Autoencoder, data: any, criterion: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    enc, dec = autoencoder(data)
    return criterion(dec, data) * scale_reconstruction_loss


def adjacent_distance_handling(autoencoder: Autoencoder, adjacent_sample_size: int,
                               scale_adjacent_distance_loss: float) -> tuple[torch.Tensor, float]:
    """
    Keeps adjacent pairs close to each other
    """
    adjacent_distance_loss = torch.tensor(0.0)
    average_distance = 0
    sampled_pairs = storage.sample_adjacent_datapoints(adjacent_sample_size)
    for pair in sampled_pairs:
        # keep adjacent close to each other
        data_point1 = storage.get_data_tensor_by_name(pair["start"])
        data_point2 = storage.get_data_tensor_by_name(pair["end"])

        encoded_i = autoencoder.encoder(data_point1.unsqueeze(0))
        encoded_j = autoencoder.encoder(data_point2.unsqueeze(0))

        distance = torch.norm((encoded_i - encoded_j), p=2)
        average_distance += distance.item()

        adjacent_distance_loss += distance * scale_adjacent_distance_loss

    adjacent_distance_loss /= adjacent_sample_size
    average_distance /= adjacent_sample_size

    return adjacent_distance_loss, average_distance


def non_adjacent_distance_handling(autoencoder: Autoencoder, non_adjacent_sample_size: int,
                                   scale_non_adjacent_distance_loss: float, distance_factor: float = 1) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other
    """
    sampled_pairs = storage.sample_non_adjacent_datapoints(non_adjacent_sample_size)
    non_adjacent_distance_loss = torch.tensor(0.0)

    for pair in sampled_pairs:
        datapoint1 = storage.get_data_tensor_by_name(pair["start"])
        datapoint2 = storage.get_data_tensor_by_name(pair["end"])

        encoded_i = autoencoder.encoder(datapoint1.unsqueeze(0))
        encoded_j = autoencoder.encoder(datapoint2.unsqueeze(0))

        distance = torch.norm((encoded_i - encoded_j), p=2)
        expected_distance = pair["distance"] * distance_factor

        non_adjacent_distance_loss += ((distance - expected_distance) ** 2) * scale_non_adjacent_distance_loss

    non_adjacent_distance_loss /= non_adjacent_sample_size
    return non_adjacent_distance_loss


from sklearn.preprocessing import StandardScaler


def normalize_data_z_score2(data):
    """
    Normalizes the data using the StandardScaler, transforming to mean=0 and std=1
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def train_autoencoder_with_distance_constraint(autoencoder, epochs=1000):
    """
    Trains the autoencoder with 2 additional losses apart from the reconstruction loss:
    - adjacent distance loss: keeps adjacent pairs close to each other
    - non-adjacent distance loss: keeps non-adjacent pairs far from each other ( in a proportional way to the distance
    between them inferred from the data )
    """

    # parameters
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    num_epochs = epochs
    scale_reconstruction_loss = 1
    scale_adjacent_distance_loss = 0.3
    scale_non_adjacent_distance_loss = 1

    adjacent_sample_size = 52
    non_adjacent_sample_size = 224

    epoch_average_loss = 0
    reconstruction_average_loss = 0
    adjacent_average_loss = 0
    non_adjacent_average_loss = 0

    epoch_print_rate = 100

    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data()))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        adjacent_distance_loss = torch.tensor(0.0)
        non_adjacent_distance_loss = torch.tensor(0.0)

        average_distance_adjacent = 0

        enc, dec = autoencoder(train_data)
        reconstruction_loss = criterion(dec, train_data) * scale_reconstruction_loss
        reconstruction_loss.backward()

        # RECONSTRUCTION LOSS
        # reconstruction_loss = reconstruction_handling(autoencoder, train_data, criterion, scale_reconstruction_loss)
        # reconstruction_loss.backward()

        # # ADJACENT DISTANCE LOSS
        # adjacent_distance_loss, average_distance_adjacent = adjacent_distance_handling(autoencoder,
        #                                                                                adjacent_sample_size,
        #                                                                                scale_adjacent_distance_loss)
        # adjacent_distance_loss.backward()
        #
        # # NON-ADJACENT DISTANCE LOSS
        # non_adjacent_distance_loss = non_adjacent_distance_handling(autoencoder, non_adjacent_sample_size,
        #                                                             scale_non_adjacent_distance_loss,
        #                                                             distance_factor=average_distance_adjacent)
        # non_adjacent_distance_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + adjacent_distance_loss.item() + non_adjacent_distance_loss.item()

        epoch_average_loss += epoch_loss
        reconstruction_average_loss += reconstruction_loss.item()
        adjacent_average_loss += adjacent_distance_loss.item()
        non_adjacent_average_loss += non_adjacent_distance_loss.item()

        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate
            adjacent_average_loss /= epoch_print_rate
            non_adjacent_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print(f"EPOCH:{epoch}/{num_epochs}")
            print(
                f"AVERAGE LOSS:{epoch_average_loss} | RECONSTRUCTION LOSS:{reconstruction_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss}")
            print(f"average distance between adjacent: {average_distance_adjacent}")
            print("--------------------------------------------------")

    return autoencoder


def run_ai():
    autoencoder = Autoencoder()
    train_autoencoder_with_distance_constraint(autoencoder, epochs=100)
    return autoencoder


def run_tests(autoencoder):
    evaluate_reconstruction_error(autoencoder)
    avg_distance_adj, avg_distance_non_adj = evaluate_distances_between_pairs(autoencoder)
    evaluate_adjacency_properties(autoencoder, avg_distance_adj * 1.25)


def evaluate_adjacency_properties(autoencoder: Autoencoder, distance_threshold: float):
    found_adjacent_pairs = []
    false_positives = []
    true_positives = []

    really_bad_false_positives = []

    total_pairs = 0
    true_adjacent_pairs = 0
    true_non_adjacent_pairs = 0

    avg_distance = 0
    avg_distance_between_found_adjacent = 0

    for i in range(len(all_sensor_data)):
        for j in range(i + 1, len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            j_x, j_y = all_sensor_data[j][1], all_sensor_data[j][2]
            total_pairs += 1

            ixy = (i_x, i_y)
            jxy = (j_x, j_y)

            avg_distance += math.sqrt((ixy[0] - jxy[0]) ** 2 + (ixy[1] - jxy[1]) ** 2)

            i_encoded = autoencoder.encoder(sensor_data[i].unsqueeze(0))
            j_encoded = autoencoder.encoder(sensor_data[j].unsqueeze(0))
            distance = torch.norm((i_encoded - j_encoded), p=2).item()

            if abs(i_x - j_x) + abs(i_y - j_y) == 1:
                true_adjacent_pairs += 1
                # print(f"({i_x}, {i_y}) - ({j_x}, {j_y}) DISTANCE: {distance:.4f}")
            else:
                true_non_adjacent_pairs += 1
                # print(f"({i_x}, {i_y}) - ({j_x}, {j_y}) NON ADJC: {distance:.4f}")

            if distance < DISTANCE_THRESHOLD:  # it is expected that adjacent distance is about sqrt(2) at least
                avg_distance_between_found_adjacent += math.sqrt((ixy[0] - jxy[0]) ** 2 + (ixy[1] - jxy[1]) ** 2)
                found_adjacent_pairs.append((i, j))
                # print(f"({i_x}, {i_y}) - ({j_x}, {j_y})")
                if abs(i_x - j_x) + abs(i_y - j_y) > 2:
                    really_bad_false_positives.append((i, j))
                if abs(i_x - j_x) + abs(i_y - j_y) > 1:
                    false_positives.append((i, j))
                elif abs(i_x - j_x) + abs(i_y - j_y) == 1:
                    true_positives.append((i, j))

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

    print(f"Percentage of adjacent paris found: {len(true_positives) / true_adjacent_pairs * 100:.2f}%")

    print(f"Average distance between all pairs: {avg_distance / total_pairs:.4f}"
          f" and between found adjacent pairs: {avg_distance_between_found_adjacent / len(found_adjacent_pairs):.4f}")


def evaluate_distances_between_pairs(autoencoder) -> float:
    """
    Gives the average distance between connected pairs ( degree 1 ) and non-connected pairs ( degree 2, 3, 4, etc. )
    """
    adjacent_data = storage.get_adjacency_data()
    non_adjacent_data = storage.get_non_adjacent_data()

    average_adjacent_embedding_distance = 0
    average_non_adjacent_embedding_distance = 0

    for connection in adjacent_data:
        start = connection["start"]
        end = connection["end"]

        start_data = storage.get_data_tensor_by_name(start)
        end_data = storage.get_data_tensor_by_name(end)

        start_embedding = autoencoder.encoder(start_data.unsqueeze(0))
        end_embedding = autoencoder.encoder(end_data.unsqueeze(0))

        distance_from_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()
        average_adjacent_embedding_distance += distance_from_embeddings

    for connection in non_adjacent_data:
        start = connection["start"]
        end = connection["end"]

        start_data = storage.get_data_tensor_by_name(start)
        end_data = storage.get_data_tensor_by_name(end)

        start_embedding = autoencoder.encoder(start_data.unsqueeze(0))
        end_embedding = autoencoder.encoder(end_data.unsqueeze(0))

        distance_from_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()
        average_non_adjacent_embedding_distance += distance_from_embeddings

    average_adjacent_embedding_distance /= len(adjacent_data)
    print(f"Average distance between connected pairs: {average_adjacent_embedding_distance:.4f}")
    average_non_adjacent_embedding_distance /= len(non_adjacent_data)
    print(f"Average distance between non-connected pairs: {average_non_adjacent_embedding_distance:.4f}")

    return average_adjacent_embedding_distance


def evaluate_reconstruction_error(autoencoder: Autoencoder) -> None:
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
            encoder, reconstructed = autoencoder(data)
            total_error += torch.sum(torch.abs(data - reconstructed)).item()

    print(
        f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error / (nr_of_samples * 8):.4f}')


def run_lee_improved(autoencoder, all_sensor_data, sensor_data):
    # same as normal lee, but keeps a queue of current pairs, and for each one of them takes the 3 closes adjacent pairs and puts them in the queue
    # if the current pair is the target pair, the algorithm stops

    starting_coords = (2, 2)
    target_coords = (2, 12)
    start_coords_data = []
    end_coords_data = []
    banned_coords = [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]

    for i in range(len(all_sensor_data)):
        i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
        if i_x == starting_coords[0] and i_y == starting_coords[1]:
            start_coords_data = sensor_data[i].unsqueeze(0)
        if i_x == target_coords[0] and i_y == target_coords[1]:
            end_coords_data = sensor_data[i].unsqueeze(0)

    start_embedding = autoencoder.encoder(start_coords_data)
    end_embedding = autoencoder.encoder(end_coords_data)

    # take all adjacent coords
    # calculate their embeddings
    # take the closest adjacent embedding to the end embedding and "step" towards it ( as in go in that direction )
    # repeat until the closest embedding is the end embedding

    current_coords = starting_coords
    explored_coords = []

    queue = [starting_coords]

    while (current_coords != target_coords):
        current_coords = queue.pop(0)
        explored_coords.append(current_coords)
        print(f"Current coords: {current_coords}")
        if current_coords[0] == target_coords[0] and current_coords[1] == target_coords[1]:
            return

        closest_distances = [1000, 1000, 1000]
        selected_coords = [[-1, -1], [-1, -1], [-1, -1]]

        for i in range(len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            # take only adjacent
            abs_dist = abs(i_x - current_coords[0]) + abs(i_y - current_coords[1])
            if (i_x, i_y) in banned_coords:
                continue

            if 0 < abs_dist <= 2:
                i_embedding = autoencoder.encoder(sensor_data[i].unsqueeze(0))
                distance = torch.norm((i_embedding - end_embedding), p=2).item()

                if distance < closest_distances[0]:
                    closest_distances[2] = closest_distances[1]
                    closest_distances[1] = closest_distances[0]
                    closest_distances[0] = distance
                    selected_coords[2] = selected_coords[1]
                    selected_coords[1] = selected_coords[0]
                    selected_coords[0] = [i_x, i_y]
                elif distance < closest_distances[1]:
                    closest_distances[2] = closest_distances[1]
                    closest_distances[1] = distance
                    selected_coords[2] = selected_coords[1]
                    selected_coords[1] = [i_x, i_y]
                elif distance < closest_distances[2]:
                    closest_distances[2] = distance
                    selected_coords[2] = [i_x, i_y]

                # if distance < closest_distances[2]:
                #     closest_distances[2] = distance
                #     place_in_queue = True

        # queue.append(selected_coords[0])
        for selected_coord in selected_coords:
            if selected_coord not in explored_coords and selected_coord not in queue:
                queue.append(selected_coord)

        if current_coords == target_coords:
            break


def find_position_data(json_data, current_position_name):
    for item in json_data:
        if item[DATA_NAME_FIELD] == current_position_name:
            return_tensor = torch.tensor(item[DATA_SENSORS_FIELD], dtype=torch.float32)
            return return_tensor

    raise Exception(f"Could not find position data for {current_position_name}")


def find_connections(datapoint_name, connections_data):
    all_cons = []
    for connection in connections_data:
        start = connection["start"]
        end = connection["end"]
        distance = connection["distance"]
        direction = connection["direction"]
        if start == datapoint_name:
            all_cons.append((start, end, distance, direction))
        if end == datapoint_name:
            direction[0] = -direction[0]
            direction[1] = -direction[1]
            all_cons.append((end, start, distance, direction))

    return all_cons


def lee_direction_step(autoencoder, current_position_name, target_position_name, json_data, connection_data):
    # get embedding for current and target
    current_position_data = find_position_data(json_data, current_position_name)
    current_embedding = autoencoder.encoder(current_position_data.unsqueeze(0))

    target_position_data = find_position_data(json_data, target_position_name)
    target_embedding = autoencoder.encoder(target_position_data.unsqueeze(0))

    connections = find_connections(current_position_name, connection_data)
    conn_names = [connection[1] for connection in connections]
    second_degree_connections = []
    for conn_name in conn_names:
        second_degree_connections += find_connections(conn_name, connection_data)

    for connection in second_degree_connections:
        start = connection[0]
        end = connection[1]
        # if start or end is not in conn_names, we add it
        if start not in conn_names:
            conn_names.append(start)
        if end not in conn_names:
            conn_names.append(end)

    closest_point = None
    closest_distances = 1000

    for conn_name in conn_names:
        conn_data = find_position_data(json_data, conn_name)
        conn_embedding = autoencoder.encoder(conn_data.unsqueeze(0))
        distance = torch.norm((conn_embedding - target_embedding), p=2).item()

        if distance < closest_distances:
            closest_distances = distance
            closest_point = conn_name

    return closest_point


def lee_improved_direction_step(autoencoder, current_position_name, target_position_name, json_data, connection_data):
    # get embedding for current and target
    current_position_data = find_position_data(json_data, current_position_name)
    current_embedding = autoencoder.encoder(current_position_data.unsqueeze(0))

    target_position_data = find_position_data(json_data, target_position_name)
    target_embedding = autoencoder.encoder(target_position_data.unsqueeze(0))

    connections = find_connections(current_position_name, connection_data)
    conn_names = [connection[1] for connection in connections]
    second_degree_connections = []
    for conn_name in conn_names:
        second_degree_connections += find_connections(conn_name, connection_data)

    for connection in second_degree_connections:
        start = connection[0]
        end = connection[1]
        # if start or end is not in conn_names, we add it
        if start not in conn_names:
            conn_names.append(start)
        if end not in conn_names:
            conn_names.append(end)

    if current_position_name == "1_3":
        print(connections)

    if current_position_name == "1_3":
        print(conn_names)

    closest_points = [None, None, None]
    closest_distances = [1000, 1000, 1000]

    for conn_name in conn_names:
        conn_data = find_position_data(json_data, conn_name)
        conn_embedding = autoencoder.encoder(conn_data.unsqueeze(0))
        distance = torch.norm((conn_embedding - target_embedding), p=2).item()

        if distance < closest_distances[0]:
            closest_distances[2] = closest_distances[1]
            closest_distances[1] = closest_distances[0]
            closest_distances[0] = distance
            closest_points[2] = closest_points[1]
            closest_points[1] = closest_points[0]
            closest_points[0] = conn_name
        elif distance < closest_distances[1]:
            closest_distances[2] = closest_distances[1]
            closest_distances[1] = distance
            closest_points[2] = closest_points[1]
            closest_points[1] = conn_name
        elif distance < closest_distances[2]:
            closest_distances[2] = distance
            closest_points[2] = conn_name

    return closest_points


def run_lee(autoencoder, all_sensor_data, sensor_data):
    starting_coords = (3, 3)
    target_coords = (11, 10)

    start_coords_data = []
    end_coords_data = []

    for i in range(len(all_sensor_data)):
        i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
        if i_x == starting_coords[0] and i_y == starting_coords[1]:
            start_coords_data = sensor_data[i].unsqueeze(0)
        if i_x == target_coords[0] and i_y == target_coords[1]:
            end_coords_data = sensor_data[i].unsqueeze(0)

    start_embedding = autoencoder.encoder(start_coords_data)
    end_embedding = autoencoder.encoder(end_coords_data)

    # take all adjacent coords
    # calculate their embeddings
    # take the closest adjacent embedding to the end embedding and "step" towards it ( as in go in that direction )
    # repeat until the closest embedding is the end embedding

    current_coords = starting_coords
    explored_coords = []

    while (current_coords != target_coords):
        current_embedding = autoencoder.encoder(sensor_data[i].unsqueeze(0))
        closest_distance = 1000
        closest_coords = None
        for i in range(len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            # take only adjacent
            abs_dist = abs(i_x - current_coords[0]) + abs(i_y - current_coords[1])
            if abs_dist <= 2 and abs_dist > 0:
                i_embedding = autoencoder.encoder(sensor_data[i].unsqueeze(0))
                distance = torch.norm((i_embedding - end_embedding), p=2).item()
                if distance < closest_distance:
                    closest_distance = distance
                    closest_coords = (i_x, i_y)

        current_coords = closest_coords
        explored_coords.append(current_coords)
        print(f"Current coords: {current_coords}")
        if current_coords == target_coords:
            break


def run_loaded_ai():
    all_sensor_data, sensor_data = preprocess_data(CollectedDataType.Data8x8)
    # all_sensor_data, sensor_data = preprocess_data(CollectedDataType.Data15x15)

    autoencoder = load_latest_ai(AIType.Autoencoder)
    # autoencoder = load_manually_saved_ai("autoenc_8x8.pth")

    run_tests(autoencoder, all_sensor_data, sensor_data)
    # run_lee(autoencoder, all_sensor_data, sensor_data)


def run_new_ai() -> None:
    global storage
    storage.load_raw_data(CollectedDataType.Data8x8)
    storage.normalize_all_data()

    autoencoder = run_ai()
    # save_ai("autoencod_vardist", AIType.Autoencoder, autoencoder)
    run_tests(autoencoder)


def run_autoencoder() -> None:
    global storage
    storage = Storage()

    run_new_ai()
    # run_loaded_ai()


if __name__ == '__main__':
    run_autoencoder()
