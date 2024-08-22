import random

import torch

from .storage import *
from typing import List
from src.ai.data_processing.ai_data_processing import normalize_data_min_max_super


class StorageSuperset(Storage):
    """
    Storage meant to handle multiple list inside the data field
    [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] instead of [1,2,3]
    """

    def __init__(self):
        super().__init__()

    def load_raw_data_super(self, env_data_filename: str, connections_filename: str):
        self.raw_env_data = read_other_data_from_file(env_data_filename)
        self.raw_connections_data = read_other_data_from_file(connections_filename)
        self._convert_raw_data_to_map()

    _non_normalized_data = None

    def freeze_non_normalized_data(self):
        copy = self.raw_env_data.copy()
        self._non_normalized_data = [x["data"][0] for x in copy]

    def normalize_incoming_data(self, data):
        # merge _non_nomralized_data with data
        copy = self._non_normalized_data.copy()
        copy.append(data)
        normalized_data = normalize_data_min_max_super(torch.tensor(copy))
        return normalized_data[-1].tolist()

    def normalize_all_data_super(self):
        # normalizes all the data
        data = self.get_pure_sensor_data()
        normalized_data = normalize_data_min_max_super(torch.tensor(data))
        length = len(self.raw_env_data)
        normalized_data = normalized_data.tolist()

        for i in range(length):
            data_normalized = normalized_data[i]
            self.raw_env_data[i]["data"] = data_normalized
            name = self.raw_env_data[i]["name"]
            self.raw_env_data_map[name]["data"] = data_normalized

    def get_datapoint_data_selected_rotation_tensor_by_name_with_noise(self, name: str, index: int) -> torch.Tensor:
        """
        Returns a random rotation from a datapoint as
        """
        deviation = random.randint(-1, 1)
        index += deviation
        lng = len(self.raw_env_data_map[name]["data"])
        if index < 0:
            index = lng
        if index >= lng:
            index = 0
            
        return torch.tensor(self.raw_env_data_map[name]["data"][index])

    def get_datapoint_data_selected_rotation_tensor_by_name(self, name: str, index: int) -> torch.Tensor:
        """
        Returns a random rotation from a datapoint as
        """
        return torch.tensor(self.raw_env_data_map[name]["data"][index])

    def get_datapoints_real_distance(self, datapoint1: str, datapoint2: str) -> float:
        dp1 = self.raw_env_data_map[datapoint1]
        dp2 = self.raw_env_data_map[datapoint2]
        x1 = dp1["params"]["x"]
        y1 = dp1["params"]["y"]
        x2 = dp2["params"]["x"]
        y2 = dp2["params"]["y"]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_datapoint_data_random_rotation_tensor_by_name(self, name: str) -> torch.Tensor:
        """
        Returns a random rotation from a datapoint as
        """
        return torch.tensor(random.choice(self.raw_env_data_map[name]["data"]))

    def get_datapoint_data_random_rotation_tensor_by_name_and_index(self, name: str) -> any:
        """
        Returns a random rotation from a datapoint as AND the index at which it was chosen
        """
        data = self.raw_env_data_map[name]["data"]
        index = random.randint(0, len(data) - 1)
        return torch.tensor(data[index]), index
