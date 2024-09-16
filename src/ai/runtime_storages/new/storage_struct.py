from dataclasses import dataclass
from typing import List, Dict, Union, Tuple
from src.ai.runtime_storages.new.cache.cache_abstract import CacheAbstract
from src.ai.runtime_storages.new.functions.cache_functions import create_baseline_caches
from src.ai.runtime_storages.new.functions.subscriber_functions import subscribers_list_initialization
from src.ai.runtime_storages.new.types import NodeData, ConnectionData, DataAlias, OperationsAlias


@dataclass
class StorageStruct:
    raw_env_data: List[NodeData]
    raw_connections_data: List[ConnectionData]
    caches: Dict[str, CacheAbstract]
    DATA_CRUD_SUBSCRIBERS: Dict[DataAlias, any]

    def __post_init__(self):
        self.raw_env_data: List[NodeData] = []
        self.raw_connections_data: List[ConnectionData] = []
        self.caches: Dict[str, CacheAbstract] = {}
        self.DATA_CRUD_SUBSCRIBERS = {}

        subscribers_list_initialization(
            self,
            data_type=DataAlias.NODE_DATA,
        )

        subscribers_list_initialization(
            self,
            data_type=DataAlias.CONNECTIONS_DATA,
        )
        create_baseline_caches(self)

    NODE_CACHE_MAP_ID: str = "node_cache_map"


def create_storage():
    return StorageStruct([], [], {}, {})
