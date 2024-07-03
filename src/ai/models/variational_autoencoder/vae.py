from torch.optim import Adam
import torch
import torch.nn as nn
import numpy as np
from src.ai.data_processing.ai_data_processing import normalize_data_min_max
from src.modules.save_load_handlers.ai_data_handle import read_data_from_file
from src.modules.save_load_handlers.parameters import *
from src.ai.models.base_model import BaseModel
from src.ai.runtime_data_storage.storage import Storage
from src.utils import array_to_tensor
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties
from src.modules.save_load_handlers.ai_models_handle import load_latest_ai, load_manually_saved_ai, save_ai


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


def reparameterization(mean, var):
    epsilon = torch.randn_like(var)
    z = mean + var * epsilon  # reparameterization trick
    return z


class VariationalAutoencoder(BaseModel):
    def __init__(self, encoder, decoder):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward_training(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encoder(x)
        z = reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encoder(x)
        z = reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat

    def encoder_training(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_var = self.encoder(x)
        return mean, log_var

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_var = self.encoder(x)
        return mean

    def decoder_training(self, x: torch.Tensor, mean: float, log_var: torch.Tensor) -> torch.Tensor:
        z = reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat

    def decoder_inference(self, x: torch.Tensor, mean: float, log_var: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(x, mean, log_var)


def init_autoencoder_variational(input_dim, hidden_dim, latent_dim, output_dim):
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, output_dim)
    model = VariationalAutoencoder(encoder, decoder)
    return model


def train_vae_without_constraint(vae: VariationalAutoencoder, epochs: int) -> VariationalAutoencoder:
    optimizer = Adam(vae.parameters(), lr=0.01)
    num_epochs = epochs

    epoch_average_loss = 0
    epoch_print_rate = 1000

    KL_loss_scale = 0.01
    reconstruction_loss_scale = 1

    def criterion(x, x_hat, mean, log_var):
        # prints fist numbers from each
        reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
        return reconstruction_loss * reconstruction_loss_scale + KLD * KL_loss_scale

    train_data = array_to_tensor(np.array(storage.get_pure_sensor_data()))

    reconstruction_average_loss = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        reconstruction_loss = 0.0

        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        x_hat, mean, log_var = vae.forward_training(train_data)
        loss = criterion(train_data, x_hat, mean, log_var)
        loss.backward()

        reconstruction_loss += loss.item()
        reconstruction_average_loss += loss.item()

        optimizer.step()

        epoch_loss += reconstruction_loss
        epoch_average_loss += epoch_loss

        if epoch % epoch_print_rate == 0:
            epoch_average_loss /= epoch_print_rate
            reconstruction_average_loss /= epoch_print_rate

            str_epoch = f'Epoch [{epoch + 1}/{num_epochs}]'
            str_reconstruction_loss = f'Reconstruction Loss: {reconstruction_loss :.4f}'

            print(f'{str_epoch} {str_reconstruction_loss}')

            epoch_average_loss = 0
            reconstruction_average_loss = 0

    return vae


def run_tests(vae: VariationalAutoencoder):
    global storage

    evaluate_reconstruction_error(vae, storage)
    avg_distance_adj = evaluate_distances_between_pairs(vae, storage)
    evaluate_adjacency_properties(vae, storage, avg_distance_adj)


def run_new_ai() -> None:
    vae = init_autoencoder_variational(8, 16, 32, 8)
    vae = train_vae_without_constraint(vae, epochs=25000)
    save_ai("vae", AIType.VariationalAutoencoder, vae)

    run_tests(vae)


def run_loaded_ai() -> None:
    global storage
    vae = load_latest_ai(AIType.VariationalAutoencoder)
    run_tests(vae)


def run_variational_autoencoder():
    global storage
    storage = Storage()
    storage.load_raw_data(CollectedDataType.Data8x8)
    storage.normalize_all_data()

    run_new_ai()
    # run_loaded_ai()


storage: Storage = Storage()
