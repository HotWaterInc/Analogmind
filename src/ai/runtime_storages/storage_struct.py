from dataclasses import dataclass, field
from typing import List, Dict
from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.ai.runtime_storages.functions.cache_functions import create_caches_general, create_caches_specialized
from src.ai.runtime_storages.functions.subscriber_functions import subscribers_list_initialization
from src.ai.runtime_storages.types import NodeData, ConnectionData, DataAlias, CachesGeneralAlias


@dataclass
class StorageStruct:
    environment_nodes_authentic: List[NodeData] = field(default_factory=list)
    connections_data_authentic: List[ConnectionData] = field(default_factory=list)
    caches: Dict[CachesGeneralAlias, CacheAbstract] = field(default_factory=dict)
    caches_functionalities: Dict[FunctionalityAlias, CacheAbstract] = field(default_factory=dict)
    data_crud_subscribers: Dict[DataAlias, any] = field(default_factory=dict)

    def __post_init__(self):
        self.environment_nodes_authentic = []
        self.connections_data_authentic = []
        self.caches = {}
        self.data_crud_subscribers = {}

        subscribers_list_initialization(
            self,
            data_type=DataAlias.NODE_DATA_AUTHENTIC,
        )

        subscribers_list_initialization(
            self,
            data_type=DataAlias.CONNECTIONS_DATA_AUTHENTIC,
        )

        create_caches_general(self)
        create_caches_specialized(self)


def create_storage():
    return StorageStruct()
