import numpy as np
import torch
import random

from .storage_superset import *
from .storage import *
from typing import List
from src.ai.data_processing.ai_data_processing import normalize_data_min_max_super
from src.ai.models.permutor import ImprovedPermutor


class StorageSuperset2(StorageSuperset):
    """
    Storage meant to handle multiple list inside the data field
    [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] instead of [1,2,3]
    """

    permutor: ImprovedPermutor = None

    def __init__(self):
        super().__init__()

    def set_permutor(self, permutor):
        self.permutor = permutor

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

    raw_env_data_permuted: List[Dict[str, any]] = []
    raw_env_data_permuted_map: Dict[str, any] = {}

    def build_permuted_data_random_rotations(self) -> None:
        """
        Returns the data point by its name
        """
        self.raw_env_data_permuted = []

        for datapoint in self.raw_env_data:
            datapoint_copy = datapoint.copy()

            name = datapoint["name"]
            data_raw: List[List[float]] = datapoint["data"]
            length = len(data_raw)
            random_index = random.randint(0, length - 1)

            datapoint_copy["data"] = data_raw[random_index]
            self.raw_env_data_permuted.append(datapoint_copy)

        for datapoint in self.raw_env_data_permuted:
            name: str = datapoint["name"]
            self.raw_env_data_permuted_map[name] = datapoint

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
        return [datapoint["data"] for datapoint in self.raw_env_data_permuted]

    _transformed_datapoints_data: Dict[str, torch.Tensor] = {}

    def get_datapoint_data_tensor_by_name_super(self, name: str) -> torch.Tensor:
        """
        Returns the data point by its name
        """
        if name not in self._transformed_datapoints_data:
            self._transformed_datapoints_data[name] = torch.tensor(self.raw_env_data_permuted_map[name]["data"],
                                                                   dtype=torch.float32)

        return self._transformed_datapoints_data[name]
