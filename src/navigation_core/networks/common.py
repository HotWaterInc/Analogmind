from dataclasses import dataclass
from torch.optim import Optimizer
import torch
from torch import nn
from src.navigation_core.networks.abstract_types import NetworkTrainingData, NetworkTrainingParams
from src.navigation_core.networks.network_abstract import NetworkAbstract
from typing import List, Callable

from src.utils.utils import get_device


@dataclass
class Loss:
    name: str
    loss_function: Callable[[NetworkAbstract, NetworkTrainingData], torch.Tensor]
    loss_scaling_factor: float


@dataclass
class Mutation:
    name: str
    mutation_function: Callable[[NetworkTrainingData], None]
    epochs_rate: int


def handle_mutations(epoch: int, training_data: NetworkTrainingData, mutations: List[Mutation]):
    for mutation in mutations:
        if epoch % mutation.epochs_rate != 0:
            continue

        mutation.mutation_function(training_data)


def handle_losses(network: NetworkAbstract, training_data: NetworkTrainingData, losses: List[Loss], losses_dict: dict):
    for loss in losses:
        loss_tensor = loss.loss_function(network, training_data)
        losses_dict[loss.name] += loss_tensor


def training_loop(network: NetworkAbstract, training_data: NetworkTrainingData, training_params: NetworkTrainingParams,
                  losses: List[Loss], mutations: List[Mutation], optimizer: Optimizer):
    loss_average_dict = {loss.name: 0 for loss in losses}
    loss_average_epoch = 0
    for epoch in range(training_params.epochs_count):
        handle_mutations(epoch, training_data, mutations)
        optimizer.zero_grad()
        losses_dict = {loss.name: torch.tensor(0.0, device=get_device()) for loss in losses}

        handle_losses(network, training_data, losses, losses_dict)
        accumulated_loss = torch.tensor(0.0, device=get_device())
        epoch_loss = torch.tensor(0.0, device=get_device())

        for loss in losses:
            accumulated_loss += losses_dict[loss.name] * loss.loss_scaling_factor
            epoch_loss += losses_dict[loss.name]
            loss_average_dict[loss.name] += losses_dict[loss.name].item()

        accumulated_loss.backward()
        optimizer.step()

        loss_average_epoch += epoch_loss.item()

        # pretty_display(epoch % epoch_print_rate)
        #
        # if epoch % epoch_print_rate == 0 and epoch != 0:
        #     epoch_average_loss /= epoch_print_rate
        #
        #     reconstruction_average_loss /= epoch_print_rate
        #     non_adjacent_average_loss /= epoch_print_rate
        #     adjacent_average_loss /= epoch_print_rate
        #     permutation_average_loss /= epoch_print_rate
        #
        #     # Print average loss for this epoch
        #     print("")
        #     print(f"EPOCH:{epoch}/{num_epochs}")
        #     print(
        #         f"RECONSTRUCTION LOSS:{reconstruction_average_loss} | NON-ADJACENT LOSS:{non_adjacent_average_loss} | ADJACENT LOSS:{adjacent_average_loss} | PERMUTATION LOSS:{permutation_average_loss}")
        #     print(f"AVERAGE LOSS:{epoch_average_loss}")
        #     print("--------------------------------------------------")
        #
        #     if non_adjacent_average_loss < THRESHOLD_MANIFOLD_NON_ADJACENT_LOSS and permutation_average_loss < THRESHOLD_MANIFOLD_PERMUTATION_LOSS and stop_at_threshold:
        #         print(f"Stopping at epoch {epoch} with loss {epoch_average_loss} because of threshold")
        #         break
        #
        #     epoch_average_loss = 0
        #     reconstruction_average_loss = 0
        #     non_adjacent_average_loss = 0
        #     adjacent_average_loss = 0
        #     permutation_average_loss = 0
        #
        #     pretty_display_reset()
        #     pretty_display_start(epoch)
