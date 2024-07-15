import numpy as np
import torch
import random

from .storage_superset import *
from .storage import *
from typing import List
from src.ai.data_processing.ai_data_processing import normalize_data_min_max_super

import math
from scipy.stats import norm

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def build_thetas(true_theta, thetas_length):
    thetas = torch.zeros(thetas_length)
    true_theta_index = true_theta * (thetas_length - 1)
    integer_index = int(true_theta_index)

    FILL_DISTANCE = 4
    for i in range(FILL_DISTANCE):
        left_index = integer_index - i
        right_index = integer_index + i

        pdf_value = norm.pdf(left_index, loc=true_theta_index, scale=1)
        if left_index < 0:
            left_index = len(thetas) + left_index
        thetas[left_index] = pdf_value

        pdf_value = norm.pdf(right_index, loc=true_theta_index, scale=1)
        if right_index >= len(thetas):
            right_index = right_index - len(thetas)
        thetas[right_index] = pdf_value

    # Normalize thetas so the maximum value is 1
    sd = 1
    peak_value = 1 / (sd * math.sqrt(2 * math.pi))
    thetas /= peak_value
    return thetas


class StorageSuperset2(StorageSuperset):
    """
    Storage meant to handle multiple list inside the data field
    [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] instead of [1,2,3]
    """

    permutor = None

    def __init__(self):

        super().__init__()

    def set_permutor(self, permutor):
        self.permutor = permutor

    def build_permuted_data_raw_with_thetas(self) -> None:
        """
        Returns the data point by its name
        """
        for index, datapoint in enumerate(self.raw_env_data):
            name = datapoint["name"]
            data_tensor = torch.tensor(np.array(datapoint["data"]), dtype=torch.float32, device=device)
            thetas_batch = []
            rotations_count = len(data_tensor)
            for index_rot in range(rotations_count):
                theta = index_rot / rotations_count
                thetas = build_thetas(theta, 36)
                thetas_batch.append(thetas)

            thetas_batch = torch.stack(thetas_batch).to(device)

            permuted_data: torch.Tensor = self.permutor(data_tensor, thetas_batch)

            permuted_data_raw = permuted_data.tolist()
            self.raw_env_data[index]["data"] = permuted_data_raw

        # rebuilds map with new values
        self._convert_raw_data_to_map()

    def build_permuted_data_raw(self) -> None:
        """
        Returns the data point by its name
        """
        for index, datapoint in enumerate(self.raw_env_data):
            name = datapoint["name"]
            data_tensor = torch.tensor(np.array(datapoint["data"]), dtype=torch.float32)
            permuted_data: torch.Tensor = self.permutor(data_tensor)

            permuted_data_raw = permuted_data.tolist()
            self.raw_env_data[index]["data"] = permuted_data_raw

        # rebuilds map with new values
        self._convert_raw_data_to_map()

    raw_env_data_permuted_choice: List[Dict[str, any]] = []
    raw_env_data_permuted_choice_map: Dict[str, any] = {}

    def build_permuted_data_random_rotations(self) -> None:
        """
        Returns the data point by its name
        """
        self.raw_env_data_permuted_choice = []

        for datapoint in self.raw_env_data:
            datapoint_copy = datapoint.copy()

            name = datapoint["name"]
            data_raw: List[List[float]] = datapoint["data"]
            length = len(data_raw)
            random_index = random.randint(0, length - 1)

            datapoint_copy["data"] = data_raw[random_index]
            self.raw_env_data_permuted_choice.append(datapoint_copy)

        for datapoint in self.raw_env_data_permuted_choice:
            name: str = datapoint["name"]
            self.raw_env_data_permuted_choice_map[name] = datapoint

    def select_random_rotations_for_permuted_data(self):
        """
        For each datapoint in the transformed data, select a random sample to train the network on
        """
        arr = []
        for datapoint in self.raw_env_data:
            name = datapoint["name"]
            data_tensor = self._transformed_datapoints_data[name]
            length = len(data_tensor)
            random_index = random.randint(0, length - 1)
            selected_data = data_tensor[random_index]
            arr.append(selected_data.tolist())
        return arr

    def get_pure_permuted_raw_env_data(self):
        """
        Returns the data point by its name
        """
        return [datapoint["data"] for datapoint in self.raw_env_data_permuted_choice]

    def get_datapoint_data_tensor_by_name_permuted(self, name: str) -> torch.Tensor:
        """
        Returns the data point by its name
        """
        return torch.tensor(self.raw_env_data_permuted_choice_map[name]["data"], dtype=torch.float32)

    # def get_datapoint_data_tensor_by_name_permuted_random_rotation(self, name: str) -> torch.Tensor:
    #     """
    #     Returns a random arr from the datapoint data field
    #     """
    #     data = self.raw_env_data_permuted_choice_map[name]["data"]
    #     index = random.randint(0, len(data) - 1)
    #     return torch.tensor(data[index], dtype=torch.float32)
