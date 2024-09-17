from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


def eulerian_distance(x_a, y_a, x_b, y_b):
    return np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)


def tanh_all_data(self: 'StorageStruct'):
    # normalizes all the data
    data = self.nodes_get_datapoints_arrays()
    normalized_data = np.tanh(np.array(data))
    length = len(self.raw_env_data)
    for i in range(length):
        self.raw_env_data[i]["data"] = normalized_data[i]
        name = self.raw_env_data[i]["name"]
        self.raw_env_data_map[name]["data"] = normalized_data[i]


def normalize_all_data(self):
    # normalizes all the data
    data = self.nodes_get_datapoints_arrays()
    normalized_data = normalize_data_min_max(np.array(data))

    length = len(self.raw_env_data)
    for i in range(length):
        self.raw_env_data[i]["data"] = normalized_data[i]
        name = self.raw_env_data[i]["name"]
        self.raw_env_data_map[name]["data"] = normalized_data[i]
