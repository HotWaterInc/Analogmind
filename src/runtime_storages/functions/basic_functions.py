from typing import List, TYPE_CHECKING
import random
import torch

from src.runtime_storages.other.cache_functions import cache_general_get
from src.runtime_storages.functions.pure_functions import eulerian_distance

from src.runtime_storages.crud.crud_functions import update_nodes_by_index
from src.runtime_storages.general_cache.cache_nodes_map import validate_cache_nodes_map
from src.runtime_storages.general_cache.cache_nodes_indexes import \
    validate_cache_nodes_indexes
from src.runtime_storages.types import ConnectionAuthenticData, NodeAuthenticData, CacheGeneralAlias, \
    ConnectionNullData, ConnectionSyntheticData
from src.utils.utils import array_to_tensor

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


def connections_authentic_get(self: 'StorageStruct') -> List[
    any]:
    return self.connections_authentic


def nodes_get_all_names(self: 'StorageStruct') -> List[str]:
    # OPTIMIZATION: cache
    return [item["name"] for item in self.nodes_authentic]


def nodes_get_datapoints_arrays(self: 'StorageStruct') -> List[any]:
    # OPTIMIZATION: cache
    return [item["datapoints_array"] for item in self.nodes_authentic]


def connections_authentic_sample(self, sample_size: int) -> List[ConnectionAuthenticData]:
    return random.sample(
        population=self.connections_authentic,
        k=sample_size,
    )


def node_get_by_name(self: 'StorageStruct', name: str) -> NodeAuthenticData:
    cache_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    cache_map = validate_cache_nodes_map(cache_map)
    return cache_map.read(node_name=name)


def node_get_by_index(self: 'StorageStruct', index: int) -> NodeAuthenticData:
    return self.nodes_authentic[index]


def node_get_datapoint_tensor_at_index(self: 'StorageStruct', node_name: str, datapoint_index: int) -> torch.Tensor:
    # OPTIMIZATION: specialized cache to cache tensor conversions

    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    node: NodeAuthenticData = node_map.read(node_name=node_name)
    return torch.tensor(node["datapoints_array"][datapoint_index], dtype=torch.float32)


def node_get_datapoints_by_name(self: 'StorageStruct', name: str) -> list:
    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    node: NodeAuthenticData = node_map.read(node_name=name)
    return node["datapoints_array"]


def node_get_index_by_name(self: 'StorageStruct', name: str) -> int:
    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    indexes_map = cache_general_get(self, CacheGeneralAlias.NODE_INDEX_MAP)
    indexes_map = validate_cache_nodes_indexes(indexes_map)

    index = indexes_map.read(node_name=name)
    if index is None:
        raise ValueError(f"Node with name {name} not found in indexes cache")

    return index


def node_get_datapoints_tensor(self: 'StorageStruct', name: str) -> torch.Tensor:
    # OPTIMIZATION: cache for tensor conversion
    cache = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    cache = validate_cache_nodes_map(cache)

    node = cache.read(node_name=name)
    return torch.tensor(node["datapoints_array"], dtype=torch.float32)


def connection_null_get_all(self: 'StorageStruct') -> List[ConnectionNullData]:
    return [item for item in self.connections_null]


def node_get_coords_metadata(self: 'StorageStruct', name: str) -> list[float]:
    cache_node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    cache_node_map = validate_cache_nodes_map(cache_node_map)
    node = cache_node_map.read(node_name=name)
    return [node["params"]["x"], node["params"]["y"]]


def node_get_closest_to_xy(self: 'StorageStruct', target_x: float, target_y: float) -> str:
    closest_datapoint = None
    closest_distance = float('inf')

    for item in self.nodes_authentic:
        name = item["name"]
        coords = node_get_coords_metadata(self, name)
        x, y = coords
        distance = eulerian_distance(x, y, target_x, target_y)
        if distance < closest_distance:
            closest_distance = distance
            closest_datapoint = name

    return closest_datapoint


def node_get_datapoint_tensor_at_index_noisy(self: 'StorageStruct', name: str, index: int,
                                             deviation: int = 1) -> torch.Tensor:
    deviation = random.randint(-deviation, deviation)
    index += deviation
    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    lng = len(node_map[name]["data"])
    if index < 0:
        index = lng + index
    if index >= lng:
        index = index - lng

    return node_get_datapoint_tensor_at_index(self, name, index)


def get_direction_between_nodes_metadata(self: 'StorageStruct', start_node_name: str, end_node_name: str) -> list[
    float]:
    start_coords = node_get_coords_metadata(self, start_node_name)
    end_coords = node_get_coords_metadata(self, end_node_name)

    dirx = end_coords[0] - start_coords[0]
    diry = end_coords[1] - start_coords[1]

    return [dirx, diry]


def get_distance_between_nodes_metadata(self: 'StorageStruct', start_node_name: str, end_node_name: str) -> float:
    start_coords = node_get_coords_metadata(self, start_node_name)
    end_coords = node_get_coords_metadata(self, end_node_name)
    xs, ys = start_coords
    xe, ye = end_coords

    return eulerian_distance(xs, ys, xe, ye)


def node_get_datapoint_tensor_at_random(self: 'StorageStruct', name: str) -> any:
    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    data = node_map[name]["data"]
    index = random.randint(0, len(data) - 1)

    return node_get_datapoint_tensor_at_index(self, name, index)


def connections_all_get(self: 'StorageStruct') -> List[
    ConnectionAuthenticData | ConnectionSyntheticData]:
    found_connections = []

    connections_authentic = self.connections_authentic
    connections_synthetic = self.connections_synthetic
    found_connections.extend(connections_authentic)
    found_connections.extend(connections_synthetic)

    return found_connections


def _connection_reorder(connection: ConnectionAuthenticData | ConnectionSyntheticData,
                        start_name: str) -> ConnectionAuthenticData:
    if connection["start"] == start_name:
        return connection
    if connection["end"] == start_name:
        connection_copy = connection.copy()
        direction = connection_copy["direction"]
        direction[0] = -direction[0]
        direction[1] = -direction[1]

        aux = connection_copy["start"]
        connection_copy["start"] = connection_copy["end"]
        connection_copy["end"] = aux
        connection_copy["direction"] = direction

        return connection_copy


def node_get_connections_all(self: 'StorageStruct', node_name: str) -> List[
    ConnectionAuthenticData | ConnectionSyntheticData]:
    found_connections = []
    connections_authentic = self.connections_authentic
    connections_synthetic = self.connections_synthetic

    # OPTIMIZATION cache
    for connection in connections_authentic:
        start = connection["start"]
        end = connection["end"]
        if start == node_name or end == node_name:
            connection_reordered = _connection_reorder(connection, node_name)
            found_connections.append(connection_reordered)

    return found_connections


def node_get_connections_null(self: 'StorageStruct', datapoint_name: str) -> List[ConnectionNullData]:
    # OPTIMIZATION cache
    found_connections = []
    connections_data = self.connections_null

    for connection in connections_data:
        start = connection["start"]
        if start == datapoint_name:
            found_connections.append(connection)

    return found_connections


def transformation_set(self: 'StorageStruct', transformation: any):
    self.transformation = transformation


def transformation_data_apply(self: 'StorageStruct') -> None:
    nodes: List[NodeAuthenticData] = self.nodes_authentic
    transformation = self.transformation

    for index, node in enumerate(nodes):
        data_tensor = array_to_tensor(node["datapoints_array"])
        transformed_data = transformation(data_tensor).tolist()
        update_nodes_by_index(
            storage=self,
            index=index,
            new_data=transformed_data,
        )


def node_get_name_at_index(self: 'StorageStruct', index: int) -> str:
    return self.nodes_authentic[index]["name"]


def node_get_datapoints_count(self: 'StorageStruct', name: str) -> int:
    node_map = cache_general_get(self, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    node: NodeAuthenticData = node_map.read(node_name=name)
    return len(node["datapoints_array"])
