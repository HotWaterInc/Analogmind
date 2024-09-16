from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import List, Dict, Union, Tuple
import random
from typing_extensions import TypedDict
import numpy as np
from src.ai.runtime_storages.new.storage_struct import StorageStruct
from src.ai.runtime_storages.new.types import NodeData, ConnectionData, DataAlias, OperationsAlias
from src.ai.runtime_storages.new.method_decorators import crud_operation
from typing import TYPE_CHECKING


@crud_operation(data_alias=DataAlias.NODE_DATA,
                operation_alias=OperationsAlias.CREATE)
def create_node(storage: StorageStruct, data: List[List[any]], name: str, params: Dict[str, any]) -> NodeData:
    new_node = NodeData(data=data, name=name, params=params)
    storage.raw_env_data.append(new_node)
    return new_node


@crud_operation(data_alias=DataAlias.NODE_DATA,
                operation_alias=OperationsAlias.DELETE)
def delete_node(storage, name: str) -> NodeData:
    target_index = None
    for i, node in enumerate(storage.raw_env_data):
        if node["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Node with name {name} not found.")

    deleted_node = storage.raw_env_data[target_index]
    del storage.raw_env_data[target_index]
    return deleted_node


@crud_operation(data_alias=DataAlias.NODE_DATA,
                operation_alias=OperationsAlias.UPDATE)
def update_node(storage, name: str, new_data: List[List[any]] = None, new_name: str = None,
                new_params: Dict[str, any] = None) -> NodeData:
    target_index = None
    for i, node in enumerate(storage.raw_env_data):
        if node["uid"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Node with name {name} not found.")

    node = storage.raw_env_data[target_index]
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


@crud_operation(data_alias=DataAlias.CONNECTIONS_DATA,
                operation_alias=OperationsAlias.CREATE)
def create_connection(storage, start: str, end: str, distance: float, direction: List[float],
                      name: str) -> ConnectionData:
    new_connection = ConnectionData(start=start, end=end, distance=distance, direction=direction, name=name)
    storage.raw_connections_data.append(new_connection)
    return new_connection


@crud_operation(data_alias=DataAlias.CONNECTIONS_DATA,
                operation_alias=OperationsAlias.DELETE)
def delete_connection(storage, name: str) -> ConnectionData:
    target_index = None
    for i, connection in enumerate(storage.raw_connections_data):
        if connection["name"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    deleted_connection = storage.raw_connections_data[target_index]
    del storage.raw_connections_data[target_index]
    return deleted_connection


@crud_operation(data_alias=DataAlias.CONNECTIONS_DATA,
                operation_alias=OperationsAlias.UPDATE)
def update_connection(storage, name: str, new_start: str = None, new_end: str = None, new_distance: float = None,
                      new_direction: List[float] = None) -> ConnectionData:
    target_index = None
    for i, connection in enumerate(storage.raw_connections_data):
        if connection["uid"] == name:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection with name {name} not found.")

    connection = storage.raw_connections_data[target_index]
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
