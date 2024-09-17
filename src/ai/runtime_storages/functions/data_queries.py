from typing import List, Dict, TYPE_CHECKING
import numpy as np
import torch

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


def connections_get_authentic(self: StorageStruct) -> List[
    any]:
    return self.connections_data_authentic


def nodes_get_names(self: StorageStruct) -> List[str]:
    # OPTIMIZATION: cache
    return [item["name"] for item in self.environment_nodes_authentic]


def nodes_get_data(self: StorageStruct):
    # OPTIMIZATION: cache
    return [item["data"] for item in self.raw_env_data]


def connections_sample(self, sample_size: int) -> List[RawConnectionData]:
    """
    Samples a number the adjacent datapoints

    :param sample_size: the number of datapoints to sample
    """
    return np.random.choice(self.raw_connections_data, sample_size, replace=False)


def get_datapoint_by_name(self, name: str) -> RawEnvironmentData:
    """
    Returns the data point by its name
    """
    return self.raw_env_data_map[name]


def get_datapoint_by_index(self, index: int) -> RawEnvironmentData:
    """
    Returns the data point by its index
    """
    return self.raw_env_data[index]


def get_datapoint_data_tensor_by_name_and_index(self, name: str, index: int) -> torch.Tensor:
    """
    Returns the data point by its name
    """

    # return self._transformed_datapoints_data[name]
    return torch.tensor(self.raw_env_data_map[name]["data"][index], dtype=torch.float32)


def get_datapoint_data_by_name(self, name: str) -> any:
    """
    Returns the data point by its name
    """

    return self.raw_env_data_map[name]["data"]


def get_datapoint_data_tensor_index_by_name(self, name: str) -> int:
    """
    Returns the index of the datapoint in the raw env data array, by its name
    """
    if name not in self.raw_env_data_map:
        return -1

    index = self.raw_env_data.index(self.raw_env_data_map[name])

    return index


def get_datapoint_data_tensor_by_name(self, name: str) -> torch.Tensor:
    """
    Returns the data point by its name
    """

    # return self._transformed_datapoints_data[name]
    return torch.tensor(self.raw_env_data_map[name]["data"], dtype=torch.float32)


def get_all_connections_null_data(self) -> List[RawConnectionData]:
    """
    Returns the connections that have null data
    """
    return [item for item in self.raw_connections_data if item["end"] == None]


def get_datapoint_metadata_coords(self, name):
    """
    Returns the metadata coordinates of a datapoint
    """
    return [self.raw_env_data_map[name]["params"]["x"], self.raw_env_data_map[name]["params"]["y"]]
