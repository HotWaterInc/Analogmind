import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, load_latest_ai, load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from typing import List, Dict, Union
from src.utils import array_to_tensor
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.evaluation.evaluation import evaluate_reconstruction_error, evaluate_distances_between_pairs, \
    evaluate_adjacency_properties, evaluate_reconstruction_error_super, evaluate_distances_between_pairs_super, \
    evaluate_adjacency_properties_super


# from src.ai.models.permutor import ImprovedPermutor


class AutoencoderPostPermutor(BaseAutoencoderModel):
    def __init__(self):
        super(AutoencoderPostPermutor, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(16, 12),
            nn.LeakyReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(12, 16),
            nn.Tanh(),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU()
        )

    def encoder_training(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.encoder1(x)
        encoded = self.encoder2(l1)
        return encoded

    def encoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_training(x)

    def decoder_training(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.decoder1(x)
        decoded = self.decoder2(l1)
        # assumes input data from permutor is already normalized between 0 and 1
        decoded = nn.functional.normalize(decoded, p=2, dim=1)
        return decoded

    def decoder_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_training(x)

    def forward_training(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_training(x)
        decoded = self.decoder_training(encoded)
        return decoded

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_training(x)


def reconstruction_handling(autoencoder: BaseAutoencoderModel, data: List[str], criterion: any,
                            scale_reconstruction_loss: int = 1) -> torch.Tensor:
    # print(data.shape)
    enc = autoencoder.encoder_training(data)
    dec = autoencoder.decoder_training(enc)
    return criterion(dec, data) * scale_reconstruction_loss


def train_autoencoder_triple_margin(autoencoder: BaseAutoencoderModel, epochs: int) -> BaseAutoencoderModel:
    """
    Training autoencoder with 3
    """
    global storage_raw

    # PARAMETERS
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    criterion = nn.L1Loss()
    STANDARD_DISTANCE = 0.5

    # criterion triple at distance 1
    criterion_triple1 = nn.TripletMarginLoss(margin=STANDARD_DISTANCE)

    num_epochs = epochs
    scale_reconstruction_loss = 1

    epoch_average_loss = 0

    epoch_print_rate = 2500
    train_data_names = storage.get_sensor_data_names()

    storage.build_permuted_data_raw()
    train_data = array_to_tensor(np.array(storage.select_random_rotations_for_permuted_data()))
    # return

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        # RECONSTRUCTION LOSS
        reconstruction_loss = reconstruction_handling(autoencoder, train_data, criterion,
                                                      scale_reconstruction_loss)
        reconstruction_loss.backward()
        triple_loss = torch.tensor(0.0)

        length = len(train_data)
        anchors = []
        positives = []
        negatives = []

        for i in range(length):
            anchor_name = train_data_names[i]
            positive_name, negative_name = storage.sample_triplet_anchor_positive_negative(anchor_name)
            # print(anchor_name, positive_name, negative_name)

            anchor_data_index = storage.get_datapoint_data_tensor_index_by_name(anchor_name)
            positive_data_index = storage.get_datapoint_data_tensor_index_by_name(positive_name)
            negative_data_index = storage.get_datapoint_data_tensor_index_by_name(negative_name)

            anchor_data = train_data[anchor_data_index]
            positive_data = train_data[positive_data_index]
            negative_data = train_data[negative_data_index]

            anchors.append(anchor_data)
            positives.append(positive_data)
            negatives.append(negative_data)

        anchors = torch.stack(anchors).unsqueeze(1)
        positives = torch.stack(positives).unsqueeze(1)
        negatives = torch.stack(negatives).unsqueeze(1)

        anchor_out = autoencoder.encoder_training(anchors)
        positive_out = autoencoder.encoder_training(positives)
        negative_out = autoencoder.encoder_training(negatives)

        triple_loss = criterion_triple1(anchor_out, positive_out, negative_out)
        triple_loss.backward()

        optimizer.step()

        epoch_loss += reconstruction_loss.item() + triple_loss.item()
        epoch_average_loss += epoch_loss
        if epoch % epoch_print_rate == 0 and epoch != 0:
            epoch_average_loss /= epoch_print_rate

            # Print average loss for this epoch
            print(f"EPOCH:{epoch}/{num_epochs}")
            print(f"AVERAGE LOSS:{epoch_average_loss}")
            print("--------------------------------------------------")

            epoch_average_loss = 0

    return autoencoder


def run_ai():
    autoencoder = AutoencoderPostPermutor()
    train_autoencoder_triple_margin(autoencoder, epochs=25000)
    return autoencoder


def run_tests(autoencoder):
    global storage_raw

    evaluate_reconstruction_error_super(autoencoder, storage)
    avg_distance_adj = evaluate_distances_between_pairs_super(autoencoder, storage)
    evaluate_adjacency_properties_super(autoencoder, storage, avg_distance_adj)


def run_loaded_ai():
    # autoencoder = load_manually_saved_ai("autoenc_dynamic10k.pth")
    # autoencoder = load_manually_saved_ai("autoencod32_high_train.pth")
    autoencoder = load_latest_ai(AIType.Autoencoder)

    run_tests(autoencoder)


def run_new_ai() -> None:
    autoencoder = run_ai()
    save_ai("autoencod_post_permutor", AIType.Autoencoder, autoencoder)
    run_tests(autoencoder)


def run_permuted_autoencoder() -> None:
    global storage
    global permutor

    permutor = load_manually_saved_ai("permutor_final1.pth")
    storage.load_raw_data_from_others("data8x8_rotated20.json")
    storage.load_raw_data_connections_from_others("data8x8_connections.json")
    storage.normalize_all_data_super()
    storage.set_transformation(permutor)

    run_new_ai()
    # run_loaded_ai()


storage: StorageSuperset2 = StorageSuperset2()
permutor = None
