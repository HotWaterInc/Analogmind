import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import normalize_data_min_max, get_json_data

class BigAutoencoder(nn.Module):
    def __init__(self):
        super(BigAutoencoder, self).__init__()

        # Encoder
        self.layer1 = nn.Sequential(
            nn.Linear(8, 16*5),  # Reduce dimension to bottleneck
            nn.Sigmoid()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(16*5, 4),  # Reduce dimension to bottleneck
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(

            nn.Linear(4, 16*5),  # Expand back to original dimension
            nn.Sigmoid()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(16*5, 8),  # Expand back to original dimension
            nn.Sigmoid()
        )

    def encoder(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.layer3(encoded)
        decoded = self.layer4(decoded)
        return encoded, decoded


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.layer1 = nn.Sequential(
            nn.Linear(8, 16),  # Reduce dimension to bottleneck
            nn.Sigmoid()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(16, 4),  # Reduce dimension to bottleneck
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(

            nn.Linear(4, 16),  # Expand back to original dimension
            nn.Sigmoid()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(16, 8),  # Expand back to original dimension
            nn.Sigmoid()
        )

    def encoder(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.layer3(encoded)
        decoded = self.layer4(decoded)
        return encoded, decoded

def train_ai(autoencoder, train_data, num_epochs, learning_rate):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)


    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for data in train_data:
            data = data.unsqueeze(0)  # Add batch dimension

            enc, dec = autoencoder(data)
            loss = criterion(dec, data)

            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        # Print average loss for this epoch
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_data):.4f}')

    print("Training completed.")

    return autoencoder

def run_big_ai(epochs = 100):
    json_data = get_json_data('../../data.json')
    sensor_data = [item['sensor_data'] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))
    encoder_epoch = epochs

    train_data = torch.tensor(sensor_data, dtype=torch.float32)
    autoencoders = []

    super_network = BigAutoencoder()
    super_network = train_ai(super_network, train_data, num_epochs=encoder_epoch, learning_rate=0.01)
    error_super_network = evaluate_error(train_data, super_network, nr_samples=len(train_data))

    return error_super_network

def run_ai(epochs = 100, nr_encoders = 5):
    json_data = get_json_data('../../data.json')
    sensor_data = [item['sensor_data'] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))
    nr_encoders = nr_encoders
    encoder_epoch = epochs

    train_data = torch.tensor(sensor_data, dtype=torch.float32)
    autoencoders = []
    for i in range(nr_encoders):
        autoencoder = Autoencoder()
        autoencoder = train_ai(autoencoder, train_data, num_epochs=encoder_epoch, learning_rate=0.01)
        autoencoders.append(autoencoder)

    super_network = Autoencoder()
    super_network = train_ai(super_network, train_data, num_epochs=encoder_epoch*nr_encoders, learning_rate=0.01)

    error_super_network = evaluate_error(train_data, super_network, nr_samples=len(train_data))

    average_error = 0
    nr_of_samples = len(train_data)
    # error for each
    for i, autoencoder in enumerate(autoencoders):
        average_error += evaluate_error(train_data, autoencoder, nr_samples=nr_of_samples)

    average_error /= len(autoencoders)

    # error for average results

    indices = np.random.choice(len(train_data), nr_of_samples, replace=False)
    total_error = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = train_data[idx].unsqueeze(0)  # Add batch dimension

            recons = []
            for autoencoder_net in autoencoders:
                encoder, reconstructed = autoencoder_net(data)
                recons.append(reconstructed)

            reconstructed = torch.mean(torch.stack(recons), dim=0)

            total_error += torch.sum(torch.abs(data - reconstructed)).item()
    error_of_averages = total_error/(nr_of_samples*8)

    print(f'AVERAGE ERROR PER ENCODER: {average_error:.4f}')
    print(f'ERROR OF AVERAGES: {error_of_averages:.4f}')

    return average_error, error_of_averages, error_super_network


def evaluate_error(train_data, autoencoder, nr_samples):
    # print("\nEvaluation on random samples from training data:")
    nr_of_samples = nr_samples
    indices = np.random.choice(len(train_data), nr_of_samples, replace=False)
    total_error = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = train_data[idx].unsqueeze(0)  # Add batch dimension
            encoder, reconstructed = autoencoder(data)
            # print(f'Random Training Sample {i+1} - Difference: {data.numpy() - reconstructed.numpy()}')
            total_error += torch.sum(torch.abs(data - reconstructed)).item()

    average_sample_error = total_error/(nr_of_samples*8)
    # print(f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error/(nr_of_samples*8):.4f}')
    return average_sample_error

def version2():
    average_error_array = []
    error_of_averages_array = []
    super_network_error_array = []
    trials = 1
    for i in range(trials):
        average_error, error_of_averages, super_network_error = run_ai(epochs=100)
        # big_network_error = run_big_ai(epochs=100)
        # print(f"Big network error: {big_network_error:.4f}")
        # print(f"Super network error: {super_network_error:.4f}")
        # print(f"Average error: {average_error:.4f}")
        # print(f"Error of averages: {error_of_averages:.4f}")
        average_error_array.append(average_error)
        error_of_averages_array.append(error_of_averages)
        super_network_error_array.append(super_network_error)

    # average decrease of error for error of averages compared to average error
    average_error_array = np.array(average_error_array)
    error_of_averages_array = np.array(error_of_averages_array)

    x_percent = 0

    average_error_avg = np.mean(average_error_array)
    error_of_averages_avg = np.mean(error_of_averages_array)
    super_network_error_avg = np.mean(super_network_error_array)

    for i in range(trials):
        a = average_error_array[i]
        b = error_of_averages_array[i]
        x_percent += (a - b) / a * 100

    print(f"Average error: {average_error_avg:.4f}")
    print(f"Average error of averages: {error_of_averages_avg:.4f}")
    print(f"Average error of super network: {super_network_error_avg:.4f}")


    super_percent_improvement = (super_network_error_avg - error_of_averages_avg) / super_network_error_avg * 100

    print(f" Average error improvement of super network: {super_percent_improvement:.4f} %")
    print(f" Average error improvement: {x_percent / trials} %")

if __name__ == "__main__":
    version2()
