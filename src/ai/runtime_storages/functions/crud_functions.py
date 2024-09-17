from typing import List, Dict
from src.ai.runtime_storages.storage_struct import StorageStruct
from src.ai.runtime_storages.new.types import NodeData, ConnectionData, DataAlias, OperationsAlias
from src.ai.runtime_storages.new.method_decorators import trigger_crud_subscribers


@trigger_crud_subscribers(data_alias=DataAlias.NODE_DATA_AUTHENTIC,
                          operation_alias=OperationsAlias.CREATE)
def create_node(storage: StorageStruct, data: List[List[any]], name: str, params: Dict[str, any]) -> NodeData:
    new_node = NodeData(data=data, name=name, params=params)
    storage.environment_nodes_authentic.append(new_node)
    return new_node


@trigger_crud_subscribers(data_alias=DataAlias.NODE_DATA_AUTHENTIC,
                          operation_alias=OperationsAlias.DELETE)
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


@trigger_crud_subscribers(data_alias=DataAlias.NODE_DATA_AUTHENTIC,
                          operation_alias=OperationsAlias.UPDATE)
def update_node(storage, name: str, new_data: List[List[any]] = None, new_name: str = None,
                new_params: Dict[str, any] = None) -> NodeData:
    target_index = None
    for i, node in enumerate(storage.environment_nodes_authentic):
        if node["uid"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Node with name {name} not found.")

    node = storage.environment_nodes_authentic[target_index]
    if new_name is None:
        new_name = node["name"]
    if new_data is None:
        new_data = node["data"]
    if new_params is None:
        new_params = node["params"]

    node["data"] = new_data
    node["name"] = new_name
    node["params"] = new_params

    return node


@trigger_crud_subscribers(data_alias=DataAlias.CONNECTIONS_DATA_AUTHENTIC,
                          operation_alias=OperationsAlias.CREATE)
def create_connection(storage, start: str, end: str, distance: float, direction: List[float],
                      name: str) -> ConnectionData:
    new_connection = ConnectionData(start=start, end=end, distance=distance, direction=direction, name=name)
    storage.connections_data_authentic.append(new_connection)
    return new_connection


@trigger_crud_subscribers(data_alias=DataAlias.CONNECTIONS_DATA_AUTHENTIC,
                          operation_alias=OperationsAlias.DELETE)
def delete_connection(storage, name: str) -> ConnectionData:
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


@trigger_crud_subscribers(data_alias=DataAlias.CONNECTIONS_DATA_AUTHENTIC,
                          operation_alias=OperationsAlias.UPDATE)
def update_connection(storage, name: str, new_start: str = None, new_end: str = None, new_distance: float = None,
                      new_direction: List[float] = None) -> ConnectionData:
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
