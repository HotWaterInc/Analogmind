import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from modules.data_handlers.ai_models_handle import save_ai, load_latest_ai, AIType
from modules.data_handlers.ai_data_processing import normalize_data_min_max
from modules.data_handlers.ai_data_handle import read_data_array_from_file
from modules.data_handlers.parameters import CollectedDataType
from modules.data_handlers.parameters import DATA_NAME_FIELD, DATA_SENSORS_FIELD, DATA_PARAMS_FIELD


DISTANCE_THRESHOLD = 5
indices_properties = []

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Dropout(0.2)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # weight_decay is the L2 regularization term


def train_autoencoder_with_distance_constraint(autoencoder, train_data, paired_indices, non_paired_indices, all_sensor_data):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.02)

    num_epochs = 10000
    scale_reconstruction_loss = 5
    scale_adjacent_distance_loss = 0.3
    scale_non_adjacent_distance_loss = 0.3
    pair_samples_adj = 24

    epoch_average_loss = 0
    epoch_print_rate = 100

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_reconstruction_loss = 0.0

        optimizer.zero_grad()

        # Forward pass
        enc, dec = autoencoder(train_data)
        loss = criterion(dec, train_data) * scale_reconstruction_loss
        loss.backward()

        total_reconstruction_loss += loss.item()
        adjacent_distance_loss = torch.tensor(0.0)

        sampled_indices = np.random.choice(len(paired_indices), pair_samples_adj, replace=False)
        sampled_pairs = [paired_indices[i] for i in sampled_indices]
        avg_adjacent_distance = 0
        for (i, j) in sampled_pairs:
            # keep adjacent close to each other
            encoded_i = autoencoder.encoder(train_data[i].unsqueeze(0))
            encoded_j = autoencoder.encoder(train_data[j].unsqueeze(0))

            distance = torch.norm((encoded_i - encoded_j), p=2)
            adjacent_distance_loss += distance * scale_adjacent_distance_loss
            avg_adjacent_distance += distance.item()

        adjacent_distance_loss /= pair_samples_adj
        adjacent_distance_loss.backward()

        non_adjacent_distance_loss = torch.tensor(0.0)

        sampled_indices = np.random.choice(len(non_paired_indices), pair_samples_adj, replace=False)
        sampled_pairs = [non_paired_indices[i] for i in sampled_indices]
        avg_distance = 0
        avg_expected_distance = 0

        for (i, j) in sampled_pairs:
            # keep non-adjacent far from each other
            encoded_i = autoencoder.encoder(train_data[i].unsqueeze(0))
            encoded_j = autoencoder.encoder(train_data[j].unsqueeze(0))

            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            j_x, j_y = all_sensor_data[j][1], all_sensor_data[j][2]

            distance = torch.norm((encoded_i - encoded_j), p=2)
            expected_distance = (math.sqrt((i_x-j_x)**2 + (i_y-j_y)**2)) #amount of distance that should be between indexes

            avg_expected_distance += expected_distance
            avg_distance += distance.item()
            non_adjacent_distance_loss += ((distance - expected_distance) ** 2) * scale_non_adjacent_distance_loss


        if non_adjacent_distance_loss > 0:
            non_adjacent_distance_loss /= pair_samples_adj
            non_adjacent_distance_loss.backward()

        optimizer.step()
        epoch_loss += total_reconstruction_loss + adjacent_distance_loss + non_adjacent_distance_loss
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0:
            epoch_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            str_epoch = f'Epoch [{epoch + 1}/{num_epochs}]'
            str_reconstruction_loss = f'Reconstruction Loss: {total_reconstruction_loss :.4f}'
            str_adjacent_distance_loss = f'Adjacent Distance Loss: {adjacent_distance_loss :.4f}'
            str_non_adjacent_distance_loss = f'Non-Adjacent Distance Loss: {non_adjacent_distance_loss :.4f}'

            str_total_loss = f'Total Loss: {(total_reconstruction_loss + adjacent_distance_loss + non_adjacent_distance_loss):.4f}'

            print(
                f'{str_epoch} - {str_reconstruction_loss} - {str_adjacent_distance_loss} - {str_non_adjacent_distance_loss} ')
            print(
                f'AVG = {epoch_average_loss} adjacent_distance ={avg_adjacent_distance / pair_samples_adj:.4f} non_adjacent_distance = {avg_distance / pair_samples_adj:.4f} ')

            # print(f"Average distance for non-adjacent pairs: {avg_distance / pair_samples_adj:.4f}")
            # print(f"Average expected distance for non-adjacent pairs: {avg_expected_distance / pair_samples_adj:.4f}")
            # print(f"Average distance for adjacent pairs: {avg_adjacent_distance / pair_samples_adj:.4f}")

    return autoencoder


def preprocess_data(data_type):
    json_data = read_data_array_from_file(data_type)

    all_sensor_data = [[item[DATA_SENSORS_FIELD], item[DATA_PARAMS_FIELD]["i"], item[DATA_PARAMS_FIELD]["j"]] for item in json_data]
    sensor_data = [item[DATA_SENSORS_FIELD] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))
    sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
    return all_sensor_data, sensor_data



def process_adjacency_properties(all_sensor_data):
    # if indexes are adjacent in the matrix, they are paired
    for i in range(len(all_sensor_data)):
        for j in range(i + 1, len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            j_x, j_y = all_sensor_data[j][1], all_sensor_data[j][2]
            indices_properties.append((i, j, abs(i_x - j_x) + abs(i_y - j_y)))

def run_ai(all_sensor_data, sensor_data):
    paired_indices = []
    non_paired_indices = []
    length = len(indices_properties)
    for i in range(length):
        if indices_properties[i][2] == 1:
            paired_indices.append((indices_properties[i][0], indices_properties[i][1]))
        else:
            non_paired_indices.append((indices_properties[i][0], indices_properties[i][1]))

    autoencoder = Autoencoder()
    # train_autoencoder_with_distance_constraint(autoencoder, sensor_data, paired_indices, non_paired_indices, all_sensor_data)

    return autoencoder


def run_tests(autoencoder, all_sensor_data, sensor_data):
    evaluate_error(sensor_data, autoencoder)
    check_distances_for_paired_indices(all_sensor_data, autoencoder, sensor_data)
    find_all_adjacent_pairs(all_sensor_data, autoencoder, sensor_data)


def find_all_adjacent_pairs(all_sensor_data, autoencoder, sensor_data):
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

            if distance < 1.2: # it is expected that adjacent distance is about sqrt(2) at least
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
    print(f"Percentage of DISTANT false positives: {len(really_bad_false_positives) / len(found_adjacent_pairs) * 100:.2f}%")
    print(f"Percentage of true positives: {len(true_positives) / len(found_adjacent_pairs) * 100:.2f}%")

    print(f"Percentage of adjacent paris found: {len(true_positives) / true_adjacent_pairs * 100:.2f}%")

    print(f"Average distance between all pairs: {avg_distance / total_pairs:.4f}"
          f" and between found adjacent pairs: {avg_distance_between_found_adjacent / len(found_adjacent_pairs):.4f}")


def check_distances_for_paired_indices(all_sensor_data, autoencoder, sensor_data):
    adjacent_pairs = []
    non_adjacent_pairs = []

    distance_adjacent = []
    distance_non_adjacent = []
    for i in range(len(all_sensor_data)):
        for j in range(i + 1, len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            j_x, j_y = all_sensor_data[j][1], all_sensor_data[j][2]

            i_encoded = autoencoder.encoder(sensor_data[i].unsqueeze(0))
            j_encoded = autoencoder.encoder(sensor_data[j].unsqueeze(0))

            distance = torch.norm((i_encoded - j_encoded), p=2).item()

            if abs(i_x - j_x) + abs(i_y - j_y) <= 1:
                adjacent_pairs.append((i, j))
                distance_adjacent.append(distance)
            else:
                non_adjacent_pairs.append((i, j))
                distance_non_adjacent.append(distance)

    print(f"Average distance for adjacent pairs: {sum(distance_adjacent) / len(distance_adjacent):.4f}")
    print(f"Average distance for non-adjacent pairs: {sum(distance_non_adjacent) / len(distance_non_adjacent):.4f}")


def evaluate_error(train_data, autoencoder):
    print("\nEvaluation on random samples from training data:")
    nr_of_samples = 64
    indices = np.random.choice(len(train_data), nr_of_samples, replace=False)
    total_error = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = train_data[idx].unsqueeze(0)  # Add batch dimension
            encoder, reconstructed = autoencoder(data)
            # print(f'Random Training Sample {i+1} - Difference: {data.numpy() - reconstructed.numpy()}')
            total_error += torch.sum(torch.abs(data - reconstructed)).item()

    print(
        f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error / (nr_of_samples * 8):.4f}')

def run_lee_improved(autoencoder, all_sensor_data, sensor_data):
    # same as normal lee, but keeps a queue of current pairs, and for each one of them takes the 3 closes adjacent pairs and puts them in the queue
    # if the current pair is the target pair, the algorithm stops

    starting_coords = (2,2)
    target_coords = (2, 12)
    start_coords_data = []
    end_coords_data = []
    banned_coords = [(0,5), (1,5), (2,5), (3,5), (4,5), (5,5), (6,5)]

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

    while(current_coords != target_coords):
        current_coords = queue.pop(0)
        explored_coords.append(current_coords)
        print(f"Current coords: {current_coords}")
        if current_coords[0] == target_coords[0] and current_coords[1] == target_coords[1]:
            return

        closest_distances = [1000, 1000, 1000]

        selected_coords = [[-1,-1], [-1,-1], [-1,-1]]

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


def run_lee(autoencoder, all_sensor_data, sensor_data):
    starting_coords = (3,3)
    target_coords = (11,10)

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

    while(current_coords != target_coords):
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
    # run_tests(autoencoder, all_sensor_data, sensor_data)
    # run_lee(autoencoder, all_sensor_data, sensor_data)
    # run_lee_improved(autoencoder, all_sensor_data, sensor_data)



def run_new_ai():
    all_sensor_data, sensor_data = preprocess_data(CollectedDataType.Data8x8)
    process_adjacency_properties(all_sensor_data)
    autoencoder = run_ai(all_sensor_data, sensor_data)
    save_ai("autoencod1", AIType.Autoencoder, autoencoder)
    # run_tests(autoencoder, all_sensor_data, sensor_data)



if __name__ == "__main__":
    # run_new_ai()
    run_loaded_ai()
    pass
