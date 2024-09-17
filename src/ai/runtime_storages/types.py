from typing import Dict, List, TypedDict
from enum import Enum


class NodeData(TypedDict):
    name: str
    datapoints_array: List[List[any]]
    params: Dict[str, any]


class ConnectionNullData(TypedDict):
    name: str
    start: str
    distance: float
    direction: List[float]


class ConnectionSyntheticData(TypedDict):
    name: str
    start: str
    end: str
    distance: float | None
    direction: List[float] | None


class ConnectionAuthenticData(TypedDict):
    name: str
    start: str
    end: str
    distance: float
    direction: List[float]


class DataAlias(Enum):
    NODE_AUTHENTIC = "node_data"
    CONNECTIONS_AUTHENTIC = "connections_data_authentic"
    CONNECTIONS_SYNTHETIC = "connections_data_synthetic"
    CONNECTIONS_NULL = "connections_null"


class CacheGeneralAlias(Enum):
    NODE_CACHE_MAP = "node_cache_map"
    NODE_INDEX_MAP = "node_cache_map"


class OperationsAlias(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
