from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import Optimizer
import torch
from torch import nn
from src.navigation_core.networks.abstract_types import NetworkTrainingData, NetworkTrainingParams
from src.navigation_core.networks.network_abstract import NetworkAbstract
from typing import List, Callable, TypeVar

from src.utils.utils import get_device, get_console


@dataclass
class Loss:
    name: str
    loss_function: Callable[[any, any], torch.Tensor]
    loss_scaling_factor: float


T = TypeVar('T', bound=NetworkTrainingData)


@dataclass
class Mutation:
    name: str
    mutation_function: Callable[[T], None]
    epochs_rate: int


def handle_mutations(epoch: int, training_data: NetworkTrainingData, mutations: List[Mutation]):
    for mutation in mutations:
        if epoch % mutation.epochs_rate != 0:
            continue

        mutation.mutation_function(training_data)


def handle_losses(network: NetworkAbstract, training_data: NetworkTrainingData, losses: List[Loss], losses_dict: dict):
    for loss in losses:
        loss_tensor = loss.loss_function(network, training_data)
        losses_dict[loss.name] = loss_tensor


def display_losses(epoch: int, losses: dict):
    loss_display = f"[Epoch {epoch}] "
    epoch_average_value = losses["epoch"]
    loss_display += f"Epoch loss: [red]{epoch_average_value:.4f}[/red] | "
    for name, value in losses.items():
        if name == "epoch":
            continue
        loss_display += f"{name}: [red]{value:.4f}[/red] | "

    loss_display = loss_display.rstrip(" | ")

    console = get_console()
    console.print(loss_display)


def display_losses_periodically(loss_average_dict: dict, epoch_print_rate: int):
    def iterator_wrapper(iterator):
        for i, item in enumerate(iterator):
            if (i + 1) % epoch_print_rate == 0:

                for loss_name in loss_average_dict:
                    loss_average_dict[loss_name] /= epoch_print_rate

                display_losses(i, loss_average_dict)

                for loss_name in loss_average_dict:
                    loss_average_dict[loss_name] = 0

            yield item

    return iterator_wrapper


def training_loop(network: NetworkAbstract, training_data: NetworkTrainingData, training_params: NetworkTrainingParams,
                  losses: List[Loss], mutations: List[Mutation]):
    optimizer = torch.optim.Adam(network.parameters(), lr=training_params.learning_rate)

    loss_average_dict = {loss.name: 0 for loss in losses}
    loss_average_dict["epoch"] = 0

    display_losses_wrapper = display_losses_periodically(loss_average_dict, training_params.epoch_print_rate)
    iterator = display_losses_wrapper(range(training_params.epochs_count))

    for epoch in iterator:
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

        loss_average_dict["epoch"] += epoch_loss.item()
