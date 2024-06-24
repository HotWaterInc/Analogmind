import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import process_data, load_ai
import math
from autoencoder import Autoencoder



def train_direction_ai(network, input_pairs, expected_direction, autoencoder):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(network.parameters(), lr=0.02)

    num_epochs = 5000
    scale_direction_loss = 1

    epoch_average_loss = 0
    epoch_print_rate = 100

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        optimizer.zero_grad()

        # Apply the autoencoder to each sensor data pair
        state1, dec1 = autoencoder(input_pairs[:, 0, :])
        state2, dec2 = autoencoder(input_pairs[:, 1, :])

        # Get the direction from the network
        direction = network(state1, state2)

        # Compute the loss
        loss = criterion(direction, expected_direction) * scale_direction_loss
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0:
            epoch_average_loss /= epoch_print_rate
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_average_loss:.4f}')
            epoch_average_loss = 0  # Reset for the next average calculation

    return network


def run_ai(all_sensor_data, sensor_data, autoencoder):

    direction_network = DirectionNetwork()

    input_pairs = []
    expected_direction = []
    input_pairs_coords = []

    for i in range(len(all_sensor_data)):
        for j in range(len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            j_x, j_y = all_sensor_data[j][1], all_sensor_data[j][2]

            if abs(i_x - j_x) + abs(i_y - j_y) > 2:
                continue

            if abs(i_x - j_x) + abs(i_y - j_y) == 0:
                continue

            direction = [j_x - i_x, j_y - i_y]
            direction = np.array(direction)
            direction = direction / np.linalg.norm(direction)

            expected_direction.append(direction)
            input_pairs.append([sensor_data[i], sensor_data[j]])
            input_pairs_coords.append([(i_x, i_y), (j_x, j_y)])

    # Check the shape of input_pairs and print some details
    input_pairs = np.array(input_pairs)
    expected_direction = np.array(expected_direction)
    print(f"First element shape: {np.array(input_pairs[0]).shape}")

    input_pairs = np.stack(input_pairs)
    input_pairs = torch.tensor(input_pairs, dtype=torch.float32)
    expected_direction = torch.tensor(expected_direction, dtype=torch.float32)

    print(f"input pair sizes: {input_pairs.size()}")
    print(f"expected direction sizes: {expected_direction.size()}")

    train_direction_ai(direction_network, input_pairs, expected_direction, autoencoder)
    evaluate_error(input_pairs, input_pairs_coords, expected_direction, direction_network, autoencoder)

    return direction_network



# def run_tests(autoencoder, all_sensor_data, sensor_data):
#     evaluate_error(sensor_data, autoencoder)
#     check_distances_for_paired_indices(all_sensor_data, autoencoder, sensor_data)
#     find_all_adjacent_pairs(all_sensor_data, autoencoder, sensor_data)

def evaluate_error(input_pairs, input_pairs_coords, expected_direction, direction_network, autoencoder):
    print("\nEvaluation on random samples from training data:")
    nr_of_samples = 100
    indices = np.random.choice(len(input_pairs), nr_of_samples, replace=False)
    total_error = 0
    displayed_examples = 3

    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = input_pairs[idx]
            direction = expected_direction[idx]

            state1, dec1 = autoencoder(data[0].unsqueeze(0))
            state2, dec2 = autoencoder(data[1].unsqueeze(0))

            direction_output = direction_network(state1, state2)
            total_error += torch.sum(torch.abs(direction_output - direction)).item()

            if displayed_examples > 0:
                print(f'Input pairs coords: {input_pairs_coords[idx]}')
                print(f'Expected Direction: {direction}')
                print(f'Output Direction: {direction_output}')
                print(f'Random Training Sample {i+1} - Difference: {direction_output.numpy() - direction.numpy()}')
                displayed_examples -= 1

    print(f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error/(nr_of_samples):.4f}')





# def run_loaded_ai():
#     all_sensor_data, sensor_data = process_data()
#     autoencoder = load_ai('autoencoder_v1_working.pth')
#     run_tests(autoencoder, all_sensor_data, sensor_data)


def run_new_ai():
    all_sensor_data, sensor_data = process_data()
    autoencoder = load_ai('autoencoder_v1_working.pth', Autoencoder)
    direction_network = run_ai(all_sensor_data, sensor_data, autoencoder)
    # run_tests(autoencoder, all_sensor_data, sensor_data)


if __name__ == "__main__":
    run_new_ai()
    # run_loaded_ai()
    pass
