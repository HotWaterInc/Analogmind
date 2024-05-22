import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import normalize_data_min_max, parse_json_string, get_json_data

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(8, 16),  # Reduce dimension to bottleneck
            nn.Tanh()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 8),  # Expand back to original dimension
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def run_ai():
    # Load data from data.json

    json_data = get_json_data('data.json')
    sensor_data = [item['sensor_data'] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))

    # Convert to torch tensor
    train_data = torch.tensor(sensor_data, dtype=torch.float32)
    autoencoder = Autoencoder()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    # Training loop
    num_epochs = 5000
    scale = 500
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # for data in train_data:
        # data = data.unsqueeze(0)  # Add batch dimension

        # Forward pass
        enc, dec = autoencoder(train_data)
        loss = criterion(dec, train_data) * scale

        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        # Print average loss for this epoch
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_data):.4f}')

    # Additional evaluation on 5 random samples from training data
    evaluate_error(train_data, autoencoder)
    torch.save(autoencoder.state_dict(), 'autoencoder_ai1.pth')

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
            # print(f'Original data: {data}')
            # print(f'Reconstructed data: {reconstructed}')
            # print(f'difference between original and reconstructed data: {(data - reconstructed)}')

    print(f'Total error on samples: {total_error:.4f} so for each sample the average error is {total_error/(nr_of_samples*8):.4f}')


if __name__ == "__main__":
    run_ai()
