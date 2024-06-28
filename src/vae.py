import torch
import torch.nn as nn
import numpy as np
from src.utils import normalize_data_min_max, get_json_data
from torch.optim import Adam

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        h_ = self.LeakyReLU(self.FC_input3(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VariationalAutoencoder(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VariationalAutoencoder, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var

    def inference(self, x):
        mean, log_var = self.Encoder(x)
        x_hat = self.Decoder(mean)
        return x_hat, mean, log_var


class ClassicAutoencoder(nn.Module):
    def __init__(self):
        super(ClassicAutoencoder, self).__init__()

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
            nn.Linear(32, 6),
            nn.Tanh(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(6, 8),
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
        return decoded, encoded, 0 # x hat, mean logvar

    def inference(self, x):
        l1 = self.encoder1(x)
        encoded = self.encoder2(l1)
        deenc = self.encoder3(encoded)
        decoded = self.decoder(deenc)
        return decoded, encoded, 0


def init_autoencoder_variational(input_dim, hidden_dim, latent_dim, output_dim):
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, output_dim)
    autoencoder = VariationalAutoencoder(encoder, decoder)
    return autoencoder

def init_autoencoder_classic():
    autoencoder = ClassicAutoencoder()
    return autoencoder


def process_data(dataset_path='../data.json'):
    json_data = get_json_data(dataset_path)
    all_sensor_data = [[item['sensor_data'], item["i_index"], item["j_index"]] for item in json_data]
    sensor_data = [item['sensor_data'] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))
    sensor_data = torch.tensor(sensor_data, dtype=torch.float32)

    return all_sensor_data, sensor_data

def train_vae_without_constraint(autoencoder, train_data, all_sensor_data, criterion, epochs):
    optimizer = Adam(autoencoder.parameters(), lr=0.001)

    num_epochs = epochs

    epoch_average_loss = 0
    epoch_print_rate = 200

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_reconstruction_loss = 0

        optimizer.zero_grad()
        # Forward pass
        x_hat, mean, log_var = autoencoder(train_data)
        loss = criterion(train_data, x_hat, mean, log_var)
        loss.backward()
        total_reconstruction_loss += loss.item()
        optimizer.step()

        epoch_loss += total_reconstruction_loss
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0:
            epoch_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            str_epoch = f'Epoch [{epoch + 1}/{num_epochs}]'
            str_reconstruction_loss = f'Reconstruction Loss: {total_reconstruction_loss :.4f}'

            print(f'{str_epoch} {str_reconstruction_loss}')

    return autoencoder


def evaluate_error(train_data, autoencoder: VariationalAutoencoder):
    print("\nEvaluation on random samples from training data:")
    nr_of_samples = 64
    indices = np.random.choice(len(train_data), nr_of_samples, replace=False)
    total_error = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = train_data[idx].unsqueeze(0)  # Add batch dimension
            x_hat, mean, log_var = autoencoder.inference(data)
            total_error += torch.sum(torch.abs(data - x_hat)).item()

    print(
        f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error / (nr_of_samples * 8):.4f}')


def check_distances_for_paired_indices(all_sensor_data, autoencoder: VariationalAutoencoder, sensor_data):
    adjacent_pairs = []
    non_adjacent_pairs = []

    distance_adjacent = []
    distance_non_adjacent = []
    avg_distance = 0
    for i in range(len(all_sensor_data)):
        for j in range(i + 1, len(all_sensor_data)):
            i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
            j_x, j_y = all_sensor_data[j][1], all_sensor_data[j][2]

            xh, i_encoded, _ = autoencoder.inference(sensor_data[i].unsqueeze(0))
            xh, j_encoded, _ = autoencoder.inference(sensor_data[j].unsqueeze(0))

            distance = torch.norm((i_encoded - j_encoded), p=2).item()

            avg_distance += distance

            if abs(i_x - j_x) + abs(i_y - j_y) <= 1:
                adjacent_pairs.append((i, j))
                distance_adjacent.append(distance)
            else:
                non_adjacent_pairs.append((i, j))
                distance_non_adjacent.append(distance)

    print(f"Average distance for adjacent pairs: {sum(distance_adjacent) / len(distance_adjacent):.4f}")
    print(f"Average distance for non-adjacent pairs: {sum(distance_non_adjacent) / len(distance_non_adjacent):.4f}")
    print(f"Average distance for all pairs: {avg_distance / (len(distance_adjacent) + len(distance_non_adjacent)):.4f}")

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

            xh, i_encoded, _ = autoencoder.inference(sensor_data[i].unsqueeze(0))
            xh, j_encoded, _ = autoencoder.inference(sensor_data[j].unsqueeze(0))
            distance = torch.norm((i_encoded - j_encoded), p=2).item()

            avg_distance += distance

            if abs(i_x - j_x) + abs(i_y - j_y) == 1:
                true_adjacent_pairs += 1
                # print(f"({i_x}, {i_y}) - ({j_x}, {j_y}) DISTANCE: {distance:.4f}")
            else:
                true_non_adjacent_pairs += 1
                # print(f"({i_x}, {i_y}) - ({j_x}, {j_y}) NON ADJC: {distance:.4f}")

            if distance < 1.2: # it is expected that adjacent distance is about sqrt(2) at least
                avg_distance_between_found_adjacent += distance
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

def run_ai(all_sensor_data, sensor_data):

    autoencoder = init_autoencoder_variational(8, 32, 6, 8)
    # autoencoder = init_autoencoder_classic()

    def criterion_classic(x, x_hat, mean, log_var):
        # L1 loss
        return  nn.functional.l1_loss(x_hat, x)

    def criterion(x, x_hat, mean, log_var):
        # prints fist numbers from each
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
        KLD =  -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
        return reproduction_loss + KLD * 0.001

    autoencoder = train_vae_without_constraint(autoencoder, sensor_data, all_sensor_data, criterion, epochs=2500)

    evaluate_error(sensor_data, autoencoder)
    check_distances_for_paired_indices(all_sensor_data, autoencoder, sensor_data)
    find_all_adjacent_pairs(all_sensor_data, autoencoder, sensor_data)

if __name__ == "__main__":
    all_sensor_data, sensor_data = process_data("modules/data_handlers/data.json")
    run_ai(all_sensor_data, sensor_data)

