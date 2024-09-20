from src.navigation_core.networks.metric_generator.training_data_struct import MetricTrainingData
from src.navigation_core.networks.metric_network_abstract import MetricNetworkAbstract
import torch
from torch import nn

from src.utils.utils import get_device


def loss_rotations(metric_generator: MetricNetworkAbstract, training_data: MetricTrainingData) -> torch.Tensor:
    """
    Makes rotations to be mapped to the same point in the embedding space
    """
    datapoints_data = training_data.rotations_array
    datapoints_data = torch.stack(datapoints_data).to(get_device())

    original_shape = datapoints_data.shape
    to_encode = datapoints_data.view(-1, original_shape[-1])
    encoded = metric_generator.encoder_training(to_encode)
    encoded = encoded.view(original_shape[0], original_shape[1], encoded.shape[-1])
    accumulated_loss = torch.cdist(encoded, encoded, p=2).mean()

    return accumulated_loss


def loss_walking_distance(metric_generator: MetricNetworkAbstract, training_data: MetricTrainingData,
                          distance_scaling_factor: float, embedding_scaling_factor: float = 1) -> torch.Tensor:
    """
    Keeps non-adjacent pairs far from each other

    distance scaling factors accounts for the range in which MSE is calculated, helps to avoid exploding or vanishing losses

    embedding scaling factor scales the embeddings to be further apart or closer together, without actually affecting the MSE loss
        * If we want generally smaller embeddings without leading to MSE collapsing to 0, we can use this parameter
    """

    walk_start, walk_end, walk_expected_distances = next(training_data.walking_dataloader)

    walk_start = torch.stack(walk_start).to(get_device())
    walk_end = torch.stack(walk_end).to(get_device())
    walk_expected_distances = torch.stack(walk_expected_distances).to(get_device())
    walk_expected_distances = walk_expected_distances * distance_scaling_factor

    encoded_i = metric_generator.encoder_training(walk_start)
    encoded_j = metric_generator.encoder_training(walk_end)

    distances_embeddings = torch.norm(encoded_i - encoded_j, p=2, dim=1)
    # making the distance of the embeddings smaller for example, forces them to become bigger in order to match the normal distance
    distances_embeddings = distances_embeddings / embedding_scaling_factor

    # Distance scaling factor controls how far away the embeddings actually are
    # For big and small distances, it readjusts MSE to not explode or vanish
    criterion = nn.MSELoss()
    walk_distance_loss = criterion(distances_embeddings, walk_expected_distances)
    return walk_distance_loss
