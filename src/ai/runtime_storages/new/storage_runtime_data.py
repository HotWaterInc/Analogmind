from enum import Enum
from functools import wraps
from typing import List, Dict, Union, Tuple
import random
from typing_extensions import TypedDict
import numpy as np
from src.ai.runtime_storages.new.types import NodeData, ConnectionData, TypeAlias, OperationsAlias
from src.ai.runtime_storages.new.storage_decorators import crud_operation


class StorageRuntimeData:
    """
    A full system for handling efficient runtime data retrieval and CRUD operations.
    Supports general and specialized caching systems, depending on your needs.
    """

    NODE_CACHE_MAP_ID = "node_cache_map"

    def __init__(self):
        self.raw_env_data: List[NodeData] = []
        self.raw_connections_data: List[ConnectionData] = []
        self.caches = {}
        self.DATA_CRUD_SUBSCRIBERS = {}

        self.create_baseline_caches()
        self.subscribers_list_initialization(
            data_type=TypeAlias.NODE_DATA,
        )

        self.subscribers_list_initialization(
            data_type=TypeAlias.CONNECTIONS_DATA,
        )

    def create_baseline_caches(self):
        """
        Creates caches which are simple and relied upon by many functions
        """
        self.cache_map_create_new(self.NODE_CACHE_MAP_ID)

    def subscribers_list_initialization(self, data_type: TypeAlias):
        self.DATA_CRUD_SUBSCRIBERS[data_type] = {
            OperationsAlias.CREATE: [],
            OperationsAlias.UPDATE: [],
            OperationsAlias.DELETE: [],
            OperationsAlias.READ: [],
        }

    def subscribe_to_crud_operations(self, data_type: TypeAlias, operation_type: OperationsAlias, subscriber):
        self.DATA_CRUD_SUBSCRIBERS[data_type][operation_type].append(subscriber)

    def cache_map_create_new(self, name: str):
        self.caches[name] = {}

    def cache_array_create_new(self, name: str):
        self.caches[name] = np.array([])

    @crud_operation(data_alias=TypeAlias.NODE_DATA,
                    operation_alias=OperationsAlias.CREATE)
    def create_node(self, data: List[List[any]], name: str, params: Dict[str, any]) -> NodeData:
        new_node = NodeData(data=data, name=name, params=params)
        self.raw_env_data.append(new_node)
        return new_node

    @crud_operation(data_alias=TypeAlias.NODE_DATA,
                    operation_alias=OperationsAlias.DELETE)
    def delete_node(self, name: str) -> NodeData:
        target_index = None
        for i, node in enumerate(self.raw_env_data):
            if node["name"] == name:
                target_index = i
                break

        if target_index is None:
            raise ValueError(f"Node with name {name} not found.")

        deleted_node = self.raw_env_data[target_index]
        del self.raw_env_data[target_index]
        return deleted_node

    @crud_operation(data_alias=TypeAlias.NODE_DATA,
                    operation_alias=OperationsAlias.UPDATE)
    def update_node(self, name: str, new_data: List[List[any]] = None, new_name: str = None,
                    new_params: Dict[str, any] = None) -> NodeData:

        target_index = None
        for i, node in enumerate(self.raw_env_data):
            if node["uid"] == name:
                target_index = i
                break

        if target_index is None:
            raise ValueError(f"Node with name {name} not found.")

        node = self.raw_env_data[target_index]
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

    @crud_operation(data_alias=TypeAlias.CONNECTIONS_DATA,
                    operation_alias=OperationsAlias.CREATE)
    def create_connection(self, start: str, end: str, distance: float, direction: List[float],
                          name: str) -> ConnectionData:
        new_connection = ConnectionData(start=start, end=end, distance=distance, direction=direction, name=name)
        self.raw_connections_data.append(new_connection)
        return new_connection

    @crud_operation(data_alias=TypeAlias.CONNECTIONS_DATA,
                    operation_alias=OperationsAlias.DELETE)
    def delete_connection(self, name: str) -> ConnectionData:
        target_index = None
        for i, connection in enumerate(self.raw_connections_data):
            if connection["name"] == name:
                target_index = i
                break

        if target_index is None:
            raise ValueError(f"Connection with name {name} not found.")

        deleted_connection = self.raw_connections_data[target_index]
        del self.raw_connections_data[target_index]
        return deleted_connection

    @crud_operation(data_alias=TypeAlias.CONNECTIONS_DATA,
                    operation_alias=OperationsAlias.UPDATE)
    def update_connection(self, name: str, new_start: str = None, new_end: str = None, new_distance: float = None,
                          new_direction: List[float] = None) -> ConnectionData:
        target_index = None
        for i, connection in enumerate(self.raw_connections_data):
            if connection["uid"] == name:
                target_index = i
                break

        if target_index is None:
            raise ValueError(f"Connection with name {name} not found.")

        connection = self.raw_connections_data[target_index]
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
