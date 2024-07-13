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
