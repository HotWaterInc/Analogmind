from typing import List, Dict, Tuple, Any
from src.ai.runtime_storages.storage_struct import StorageStruct
from src.ai.runtime_storages.types import NodeData, ConnectionAuthenticData, DataAlias, OperationsAlias, \
    ConnectionSyntheticData
from src.ai.runtime_storages.method_decorators import trigger_create_subscribers, trigger_delete_subscribers, \
    trigger_update_subscribers


@trigger_create_subscribers(data_alias=DataAlias.NODE_AUTHENTIC,
                            )
def create_node(storage: StorageStruct, data: List[List[any]], name: str, params: Dict[str, any]) -> NodeData:
    new_node = NodeData(datapoints_array=data, name=name, params=params)
    storage.environment_nodes_authentic.append(new_node)
    return new_node


@trigger_delete_subscribers(data_alias=DataAlias.NODE_AUTHENTIC,
                            )
def delete_node(storage, name: str) -> NodeData:
    target_index = None
    for i, node in enumerate(storage.environment_nodes_authentic):
        if node["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Node with name {name} not found.")

    deleted_node = storage.environment_nodes_authentic[target_index]
    del storage.environment_nodes_authentic[target_index]
    return deleted_node


@trigger_create_subscribers(data_alias=DataAlias.NODE_AUTHENTIC,
                            )
def update_node(storage, name: str, new_data: List[List[any]] = None, new_name: str = None,
                new_params: Dict[str, any] = None) -> tuple[Any, Any]:
    target_index = None
    for i, node in enumerate(storage.environment_nodes_authentic):
        if node["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Node with name {name} not found.")

    node = storage.environment_nodes_authentic[target_index]
    old_node = node.copy()
    if new_name is None:
        new_name = node["name"]
    if new_data is None:
        new_data = node["data"]
    if new_params is None:
        new_params = node["params"]

    node["data"] = new_data
    node["name"] = new_name
    node["params"] = new_params
    new_node = node

    return old_node, new_node


@trigger_create_subscribers(data_alias=DataAlias.CONNECTIONS_AUTHENTIC,
                            )
def create_connection_authentic(storage, start: str, end: str, distance: float, direction: List[float],
                                name: str) -> ConnectionAuthenticData:
    new_connection = ConnectionAuthenticData(start=start, end=end, distance=distance, direction=direction, name=name)
    storage.connections_data_authentic.append(new_connection)
    return new_connection


@trigger_delete_subscribers(data_alias=DataAlias.CONNECTIONS_AUTHENTIC,
                            )
def delete_connection_authentic(storage, name: str) -> ConnectionAuthenticData:
    target_index = None
    for i, connection in enumerate(storage.connections_data_authentic):
        if connection["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    deleted_connection = storage.connections_data_authentic[target_index]
    del storage.connections_data_authentic[target_index]
    return deleted_connection


@trigger_update_subscribers(data_alias=DataAlias.CONNECTIONS_AUTHENTIC,
                            )
def update_connection_authentic(storage, name: str, new_start: str = None, new_end: str = None,
                                new_distance: float = None,
                                new_direction: List[float] = None) -> ConnectionAuthenticData:
    target_index = None
    for i, connection in enumerate(storage.connections_data_authentic):
        if connection["uid"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    connection = storage.connections_data_authentic[target_index]
    if new_start is None:
        new_start = connection["start"]
    if new_end is None:
        new_end = connection["end"]
    if new_distance is None:
        new_distance = connection["distance"]
    if new_direction is None:
        new_direction = connection["direction"]

    connection["start"] = new_start
    connection["end"] = new_end
    connection["distance"] = new_distance
    connection["direction"] = new_direction

    return connection


@trigger_create_subscribers(data_alias=DataAlias.CONNECTIONS_SYNTHETIC)
def create_connection_synthetic(storage, start: str, end: str, name: str) -> ConnectionSyntheticData:
    new_connection = ConnectionSyntheticData(start=start, end=end, distance=None, direction=None, name=name)
    storage.connections_data_synthetic.append(new_connection)
    return new_connection


@trigger_delete_subscribers(data_alias=DataAlias.CONNECTIONS_SYNTHETIC)
def delete_connection_synthetic(storage, name: str) -> ConnectionSyntheticData:
    target_index = None
    for i, connection in enumerate(storage.connections_data_synthetic):
        if connection["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    deleted_connection = storage.connections_data_synthetic[target_index]
    del storage.connections_data_synthetic[target_index]
    return deleted_connection


@trigger_update_subscribers(data_alias=DataAlias.CONNECTIONS_SYNTHETIC)
def update_connection_synthetic(storage, name: str, new_start: str = None,
                                new_end: str = None) -> ConnectionSyntheticData:
    target_index = None
    for i, connection in enumerate(storage.connections_data_synthetic):
        if connection["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    connection = storage.connections_data_synthetic[target_index]
    if new_start is None:
        new_start = connection["start"]
    if new_end is None:
        new_end = connection["end"]

    connection["start"] = new_start
    connection["end"] = new_end

    return connection


@trigger_create_subscribers(data_alias=DataAlias.CONNECTIONS_NULL)
def create_connection_null(storage, start: str, end: str, distance: float, direction: List[float],
                           name: str) -> ConnectionAuthenticData:
    new_connection = ConnectionAuthenticData(start=start, end=end, distance=distance, direction=direction, name=name)
    storage.connections_data_null.append(new_connection)
    return new_connection


@trigger_delete_subscribers(data_alias=DataAlias.CONNECTIONS_NULL)
def delete_connection_null(storage, name: str) -> ConnectionAuthenticData:
    target_index = None
    for i, connection in enumerate(storage.connections_data_null):
        if connection["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    deleted_connection = storage.connections_data_null[target_index]
    del storage.connections_data_null[target_index]
    return deleted_connection


@trigger_update_subscribers(data_alias=DataAlias.CONNECTIONS_NULL)
def update_connection_null(storage, name: str, new_start: str = None, new_end: str = None, new_distance: float = None,
                           new_direction: List[float] = None) -> ConnectionAuthenticData:
    target_index = None
    for i, connection in enumerate(storage.connections_data_null):
        if connection["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    connection = storage.connections_data_null[target_index]
    if new_start is None:
        new_start = connection["start"]
    if new_end is None:
        new_end = connection["end"]
    if new_distance is None:
        new_distance = connection["distance"]
    if new_direction is None:
        new_direction = connection["direction"]

    connection["start"] = new_start
    connection["end"] = new_end
    connection["distance"] = new_distance
    connection["direction"] = new_direction

    return connection
