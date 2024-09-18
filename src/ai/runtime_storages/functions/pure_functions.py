from typing import TYPE_CHECKING
import random
import numpy as np

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


def eulerian_distance(x_a, y_a, x_b, y_b) -> float:
    return np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)


def noise_add_with_overflow(number: int, lower_bound: int, upper_bound: int, deviation: int = 1) -> int:
    """
    Adds noise and overflows from lower to upper bound and from upper to lower bound
    """
    offset = random.randint(-deviation, deviation)
    noisy_number = number + offset

    if noisy_number < lower_bound:
        return upper_bound - (lower_bound - noisy_number)
    elif noisy_number > upper_bound:
        return lower_bound + (noisy_number - upper_bound)
