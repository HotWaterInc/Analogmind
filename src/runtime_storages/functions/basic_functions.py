from typing import List, TYPE_CHECKING
import random
import torch
from src.navigation_core.autonomous_exploration.params import IS_CLOSE_THRESHOLD
from src.navigation_core.pure_functions import calculate_coords_distance, connection_reverse_order
from src.runtime_storages.functions.pure_functions import eulerian_distance
from src.runtime_storages.crud.crud_functions import update_nodes_by_index
from src.runtime_storages.general_cache.cache_nodes_map import validate_cache_nodes_map
from src.runtime_storages.general_cache.cache_nodes_indexes import \
    validate_cache_nodes_indexes
from src.runtime_storages.other import cache_general_get
from src.runtime_storages.types import ConnectionAuthenticData, NodeAuthenticData, CacheGeneralAlias, \
    ConnectionNullData, ConnectionSyntheticData
from src.utils.utils import array_to_tensor

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


def connections_synthetic_get(storage: 'StorageStruct') -> List[
    ConnectionSyntheticData]:
    return storage.connections_synthetic


def connections_authentic_get(storage: 'StorageStruct') -> List[
    ConnectionAuthenticData]:
    return storage.connections_authentic


def nodes_get_all_names(storage: 'StorageStruct') -> List[str]:
    # OPTIMIZATION: cache
    return [item["name"] for item in storage.nodes_authentic]


def nodes_get_all(storage: 'StorageStruct') -> List[NodeAuthenticData]:
    return storage.nodes_authentic


def nodes_get_datapoints_arrays(storage: 'StorageStruct') -> List[any]:
    # OPTIMIZATION: cache
    return [item["datapoints_array"] for item in storage.nodes_authentic]


def connections_authentic_sample(storage, sample_size: int) -> List[ConnectionAuthenticData]:
    return random.sample(
        population=storage.connections_authentic,
        k=sample_size,
    )


def node_get_by_name(storage: 'StorageStruct', name: str) -> NodeAuthenticData:
    cache_map = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    cache_map = validate_cache_nodes_map(cache_map)
    return cache_map.read(node_name=name)


def node_get_by_index(storage: 'StorageStruct', index: int) -> NodeAuthenticData:
    return storage.nodes_authentic[index]


def node_get_datapoint_tensor_at_index(storage: 'StorageStruct', node_name: str, datapoint_index: int) -> torch.Tensor:
    # OPTIMIZATION: specialized cache to cache tensor conversions

    node_map = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    node: NodeAuthenticData = node_map.read(node_name=node_name)
    return torch.tensor(node["datapoints_array"][datapoint_index], dtype=torch.float32)


def node_get_datapoints_by_name(storage: 'StorageStruct', name: str) -> list:
    node_map = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    node: NodeAuthenticData = node_map.read(node_name=name)
    return node["datapoints_array"]


def node_get_index_by_name(storage: 'StorageStruct', name: str) -> int:
    indexes_map = cache_general_get(storage, CacheGeneralAlias.NODE_INDEX_MAP)
    indexes_map = validate_cache_nodes_indexes(indexes_map)

    index = indexes_map.read(node_name=name)
    if index is None:
        raise ValueError(f"Node with name {name} not found in indexes cache")

    return index


def node_get_datapoints_tensor(storage: 'StorageStruct', name: str) -> torch.Tensor:
    # OPTIMIZATION: cache for tensor conversion
    cache = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    cache = validate_cache_nodes_map(cache)

    node = cache.read(node_name=name)
    return torch.tensor(node["datapoints_array"], dtype=torch.float32)


def connection_null_get_all(storage: 'StorageStruct') -> List[ConnectionNullData]:
    return [item for item in storage.connections_null]


def node_get_coords_metadata(storage: 'StorageStruct', name: str) -> list[float]:
    cache_node_map = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    cache_node_map = validate_cache_nodes_map(cache_node_map)
    node = cache_node_map.read(node_name=name)
    return [node["params"]["x"], node["params"]["y"]]


def node_get_closest_to_xy(storage: 'StorageStruct', target_x: float, target_y: float) -> str:
    closest_datapoint = None
    closest_distance = float('inf')

    for item in storage.nodes_authentic:
        name = item["name"]
        coords = node_get_coords_metadata(storage, name)
        x, y = coords
        distance = eulerian_distance(x, y, target_x, target_y)
        if distance < closest_distance:
            closest_distance = distance
            closest_datapoint = name

    return closest_datapoint


def node_get_datapoint_tensor_at_index_noisy(storage: 'StorageStruct', name: str, index: int,
                                             deviation: int = 1) -> torch.Tensor:
    deviation = random.randint(-deviation, deviation)
    index += deviation
    node_map = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    lng = len(node_map[name]["data"])
    if index < 0:
        index = lng + index
    if index >= lng:
        index = index - lng

    return node_get_datapoint_tensor_at_index(storage, name, index)


def get_direction_between_nodes_metadata(storage: 'StorageStruct', start_node_name: str, end_node_name: str) -> list[
    float]:
    start_coords = node_get_coords_metadata(storage, start_node_name)
    end_coords = node_get_coords_metadata(storage, end_node_name)

    dirx = end_coords[0] - start_coords[0]
    diry = end_coords[1] - start_coords[1]

    return [dirx, diry]


def get_distance_between_nodes_metadata(storage: 'StorageStruct', start_node_name: str, end_node_name: str) -> float:
    start_coords = node_get_coords_metadata(storage, start_node_name)
    end_coords = node_get_coords_metadata(storage, end_node_name)
    xs, ys = start_coords
    xe, ye = end_coords

    return eulerian_distance(xs, ys, xe, ye)


def node_get_datapoint_tensor_at_random(storage: 'StorageStruct', name: str) -> any:
    node_map = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    data = node_map.read(name)["datapoints_array"]
    index = random.randint(0, len(data) - 1)

    return node_get_datapoint_tensor_at_index(storage, name, index)


def connections_all_get(storage: 'StorageStruct') -> List[
    ConnectionAuthenticData | ConnectionSyntheticData]:
    found_connections = []

    connections_authentic = storage.connections_authentic
    connections_synthetic = storage.connections_synthetic
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


def node_get_connections_all(storage: 'StorageStruct', node_name: str) -> List[
    ConnectionAuthenticData | ConnectionSyntheticData]:
    found_connections = []
    connections_authentic = storage.connections_authentic
    connections_synthetic = storage.connections_synthetic
    all_connections = []
    all_connections.extend(connections_authentic)
    all_connections.extend(connections_synthetic)

    # OPTIMIZATION cache
    for connection in all_connections:
        start = connection["start"]
        end = connection["end"]
        if start == node_name or end == node_name:
            connection_reordered = _connection_reorder(connection, node_name)
            found_connections.append(connection_reordered)

    return found_connections


def node_get_connections_null(storage: 'StorageStruct', datapoint_name: str) -> List[ConnectionNullData]:
    # OPTIMIZATION cache
    found_connections = []
    connections_data = storage.connections_null

    for connection in connections_data:
        start = connection["start"]
        if start == datapoint_name:
            found_connections.append(connection)

    return found_connections


def transformation_set(storage: 'StorageStruct', transformation: any):
    storage.transformation = transformation


def transformation_data_apply(storage: 'StorageStruct') -> None:
    nodes: List[NodeAuthenticData] = storage.nodes_authentic
    transformation = storage.transformation

    for index, node in enumerate(nodes):
        data_tensor = array_to_tensor(node["datapoints_array"])
        transformed_data = transformation(data_tensor).tolist()
        update_nodes_by_index(
            storage=storage,
            index=index,
            new_data=transformed_data,
        )


def node_get_name_at_index(storage: 'StorageStruct', index: int) -> str:
    return storage.nodes_authentic[index]["name"]


def node_get_datapoints_count(storage: 'StorageStruct', name: str) -> int:
    node_map = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    node_map = validate_cache_nodes_map(node_map)
    node: NodeAuthenticData = node_map.read(node_name=name)
    return len(node["datapoints_array"])


def check_node_is_known_from_metadata(storage: 'StorageStruct', current_coords: list[float]) -> bool:
    """
    Check if the current coordinates are close to any known
    """
    nodes_names = nodes_get_all_names(storage)
    for name in nodes_names:
        coords = node_get_coords_metadata(storage, name)
        calculate_coords_distance(coords, current_coords)
        if calculate_coords_distance(coords, current_coords) < IS_CLOSE_THRESHOLD:
            return True

    return False


def node_get_connections_adjacent(storage: 'StorageStruct', node_name: str) -> List[
    ConnectionAuthenticData | ConnectionSyntheticData]:
    # OPTIMIZATION cache

    found_connections = []
    total_connections = connections_all_get(storage)

    for connection in total_connections:
        start = connection["start"]
        end = connection["end"]

        if start == node_name:
            found_connections.append(connection.copy())
        elif end == node_name:
            connection_copy = connection.copy()
            reversed_connection = connection_reverse_order(connection_copy)
            found_connections.append(reversed_connection)

    return found_connections


def connections_authentic_check_if_exists(storage: 'StorageStruct', start: str, end: str) -> bool:
    for connection in storage.connections_authentic:
        if connection["start"] == start and connection["end"] == end:
            return True

    return False


def connections_synthetic_check_if_exists(storage: 'StorageStruct', start: str, end: str) -> bool:
    for connection in storage.connections_synthetic:
        if connection["start"] == start and connection["end"] == end:
            return True

    return False


def connections_classify_into_authentic_synthetic(storage: 'StorageStruct', connections: List[
    ConnectionAuthenticData | ConnectionSyntheticData]) -> tuple[
    List[ConnectionAuthenticData], List[ConnectionSyntheticData]]:
    authentic = []
    synthetic = []

    for connection in connections:
        start = connection["start"]
        end = connection["end"]
        check_authentic = connections_authentic_check_if_exists(storage, start, end)
        check_synthetic = connections_synthetic_check_if_exists(storage, start, end)

        if check_authentic and check_synthetic:
            raise ValueError(f"Connection {start} -> {end} is both authentic and synthetic")

        if check_authentic:
            authentic.append(connection)

        if check_synthetic:
            synthetic.append(connection)

    return authentic, synthetic
