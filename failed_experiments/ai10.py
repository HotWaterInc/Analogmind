import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import normalize_data_min_max, get_json_data

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(8, 4),  # Reduce dimension to bottleneck
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),  # Expand back to original dimension
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_ai(autoencoder, train_data, num_epochs, learning_rate=0.02):
    criterion = nn.MSELoss()
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_data):.4f}')

    print("Training completed.")

    return autoencoder

def run_ai():
    json_data = get_json_data('../data.json')
    sensor_data = [item['sensor_data'] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))

    train_data = torch.tensor(sensor_data, dtype=torch.float32)
    autoencoders = []
    nr_encoders = 20
    for i in range(nr_encoders):
        autoencoder = Autoencoder()
        autoencoder = train_ai(autoencoder, train_data, num_epochs=200, learning_rate=0.01)
        autoencoders.append(autoencoder)

    error = 0
    nr_of_samples = 20
    # error for each
    for i, autoencoder in enumerate(autoencoders):
        error += evaluate_error(train_data, autoencoder, nr_samples=nr_of_samples)

    error /= len(autoencoders)

    # error for average results
    print("AVERAGE ERR RESULT ")
    print(error)

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

    print(f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error/(nr_of_samples*8):.4f}')


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

if __name__ == "__main__":
    run_ai()
