from typing import Dict, List, TypedDict
from enum import Enum


class NodeData(TypedDict):
    name: str
    data: List[List[any]]
    params: Dict[str, any]


class ConnectionData(TypedDict):
    name: str
    start: str
    end: str
    distance: float
    direction: List[float]


class DataAlias(Enum):
    NODE_DATA_AUTHENTIC = "node_data"
    CONNECTIONS_DATA_AUTHENTIC = "connections_data"


class CachesGeneralAlias(Enum):
    NODE_CACHE_MAP_ID = "node_cache_map"


class OperationsAlias(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
