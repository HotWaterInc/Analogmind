from typing import List, Dict, TYPE_CHECKING
import random
import numpy as np
import torch
from src.ai.runtime_storages.functions.cache_functions import cache_general_get
from src.ai.runtime_storages.functions.pure_math import eulerian_distance
from src.ai.runtime_storages.types import ConnectionAuthenticData, NodeData, CacheGeneralAlias, ConnectionNullData

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


def connections_authentic_get(self: 'StorageStruct') -> List[
    any]:
    return self.connections_data_authentic


def nodes_get_all_names(self: 'StorageStruct') -> List[str]:
    # OPTIMIZATION: cache
    return [item["name"] for item in self.environment_nodes_authentic]


def nodes_get_datapoints_arrays(self: 'StorageStruct') -> List[any]:
    # OPTIMIZATION: cache
    return [item["datapoints_array"] for item in self.environment_nodes_authentic]


def connections_authentic_sample(self, sample_size: int) -> List[ConnectionAuthenticData]:
    return random.sample(
        population=self.connections_data_authentic,
        k=sample_size,
    )


def node_get_by_name(self: 'StorageStruct', name: str) -> NodeData:
    cache_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    return cache_map[name]


def node_get_by_index(self: 'StorageStruct', index: int) -> NodeData:
    return self.environment_nodes_authentic[index]


def node_get_datapoint_tensor_at_index(self: 'StorageStruct', node_name: str, sample_index: int) -> torch.Tensor:
    # OPTIMIZATION: specialized cache to cache tensor conversions

    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    return torch.tensor(node_map[node_name]["data"][sample_index], dtype=torch.float32)


def node_get_datapoints_by_name(self: 'StorageStruct', name: str) -> any:
    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    return node_map[name]["data"]


def node_get_index_by_name(self: 'StorageStruct', name: str) -> int:
    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    indexes_map = cache_general_get(self, CacheGeneralAlias.NODE_INDEX_MAP)
    if name not in indexes_map.cache_map:
        raise ValueError(f"Node with name {name} not found in indexes map")

    index = indexes_map.cache_map[name]
    return index


def node_get_datapoints_tensor(self: 'StorageStruct', name: str) -> torch.Tensor:
    # OPTIMIZATION: cache for tensor conversion
    cache = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    return torch.tensor(cache[name]["data"], dtype=torch.float32)


def connection_null_get_all(self: 'StorageStruct') -> List[ConnectionNullData]:
    return [item for item in self.connections_data_null]


def node_get_metadata_coords(self: 'StorageStruct', name: str):
    cache_node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    return [cache_node_map[name]["params"]["x"], cache_node_map[name]["params"]["y"]]


def node_get_closest_to_xy(self: 'StorageStruct', target_x: float, target_y: float) -> str:
    closest_datapoint = None
    closest_distance = float('inf')

    for item in self.environment_nodes_authentic:
        name = item["name"]
        coords = node_get_metadata_coords(self, name)
        x, y = coords
        distance = eulerian_distance(x, y, target_x, target_y)
        if distance < closest_distance:
            closest_distance = distance
            closest_datapoint = name

    return closest_datapoint
