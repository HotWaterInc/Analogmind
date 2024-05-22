import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import normalize_data_min_max, parse_json_string, get_json_data
import math

DISTANCE_THRESHOLD = 5


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(8, 16),  # Reduce dimension to bottleneck
            nn.ReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(16, 32),  # Reduce dimension to bottleneck
            nn.LeakyReLU()
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(32, 16),  # Reduce dimension to bottleneck
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 8),  # Expand back to original dimension
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


def train_autoencoder_with_distance_constraint(autoencoder, train_data, paired_indices, non_paired_indices, all_sensor_data):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.02)

    num_epochs = 20000
    scale_reconstruction_loss = 5
    scale_adjacent_distance_loss = 0.3
    scale_non_adjacent_distance_loss = 0.3
    pair_samples_adj = 52

    epoch_average_loss = 0
    epoch_print_rate = 1000

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_reconstruction_loss = 0.0

        # if epoch % 200 == 0 and scale_non_adjacent_distance_loss < 100:
        #     scale_non_adjacent_distance_loss += 10
        #     # print(f"Scale non-adjacent distance loss: {scale_non_adjacent_distance_loss}")

        # if epoch % 100 == 0 and scale_adjacent_distance_loss < 5:
        #     scale_adjacent_distance_loss += 1
        #     # scale_non_adjacent_distance_loss += 1


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

            adjacent_distance_loss += torch.norm((encoded_i - encoded_j) * 2, p=2) * scale_adjacent_distance_loss
            avg_adjacent_distance += torch.norm((encoded_i - encoded_j), p=2).item()
            # print(f"({all_sensor_data[i][1]}, {all_sensor_data[i][2]}) - ({all_sensor_data[j][1]}, {all_sensor_data[j][2]}) DISTANCE: {torch.norm((encoded_i - encoded_j), p=2).item()}")

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
            expected_distance = (math.sqrt((i_x-j_x)**2 + (i_y-j_y)**2)) # amount of distance that should be between indexes

            avg_expected_distance += expected_distance
            avg_distance += distance.item()
            # DISTANCE_THRESHOLD = 5
            # if distance < DISTANCE_THRESHOLD:
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


def process_data():
    json_data = get_json_data('data.json')
    all_sensor_data = [[item['sensor_data'], item["i_index"], item["j_index"]] for item in json_data]
    sensor_data = [item['sensor_data'] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))
    sensor_data = torch.tensor(sensor_data, dtype=torch.float32)

    return all_sensor_data, sensor_data


def load_autoencoder(name):
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(name))
    return autoencoder


def run_ai(all_sensor_data, sensor_data):
    # if indexes are adjacent in the matrix, they are paired
    paired_indices = []
    non_paired_indices = []
    for i in range(len(all_sensor_data)):
        for j in range(i + 1, len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            j_x, j_y = all_sensor_data[j][1], all_sensor_data[j][2]

            if abs(i_x - j_x) + abs(i_y - j_y) <= 1:
                paired_indices.append((i, j))
            elif abs(i_x - j_x) + abs(i_y - j_y) > 1:
                non_paired_indices.append((i, j))

    autoencoder = Autoencoder()

    train_autoencoder_with_distance_constraint(autoencoder, sensor_data, paired_indices, non_paired_indices, all_sensor_data)

    torch.save(autoencoder.state_dict(), 'autoencoder_v2.pth')
    return autoencoder


def run_tests(autoencoder, all_sensor_data, sensor_data):
    evaluate_error(sensor_data, autoencoder)
    check_distances_for_paired_indices(all_sensor_data, autoencoder, sensor_data)
    find_all_adjacent_pairs(all_sensor_data, autoencoder, sensor_data)


def find_all_adjacent_pairs(all_sensor_data, autoencoder, sensor_data ):
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


def run_loaded_ai():
    all_sensor_data, sensor_data = process_data()
    autoencoder = load_autoencoder('autoencoder_v1_working.pth')
    run_tests(autoencoder, all_sensor_data, sensor_data)


def run_new_ai():
    all_sensor_data, sensor_data = process_data()
    autoencoder = run_ai(all_sensor_data, sensor_data)
    run_tests(autoencoder, all_sensor_data, sensor_data)


if __name__ == "__main__":
    # run_new_ai()
    run_loaded_ai()
    pass
