from dataclasses import dataclass, field
from typing import List, Dict
from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.ai.runtime_storages.functions.cache_functions import create_caches_general, create_caches_specialized
from src.ai.runtime_storages.functions.subscriber_functions import subscribers_list_initialization
from src.ai.runtime_storages.types import NodeData, ConnectionAuthenticData, DataAlias, CacheGeneralAlias, \
    ConnectionSyntheticData, ConnectionNullData


@dataclass
class StorageStruct:
    environment_nodes_authentic: List[NodeData] = field(default_factory=list)

    connections_data_authentic: List[ConnectionAuthenticData] = field(default_factory=list)
    connections_data_synthetic: List[ConnectionSyntheticData] = field(default_factory=list)
    connections_data_null: List[ConnectionNullData] = field(default_factory=list)

    caches: Dict[CacheGeneralAlias, CacheAbstract] = field(default_factory=dict)
    caches_functionalities: Dict[FunctionalityAlias, CacheAbstract] = field(default_factory=dict)
    data_crud_subscribers: Dict[DataAlias, any] = field(default_factory=dict)

    def __post_init__(self):
        self.environment_nodes_authentic = []
        self.connections_data_authentic = []
        self.connections_data_synthetic = []
        self.connections_data_null = []
        self.caches = {}
        self.data_crud_subscribers = {}

        subscribers_list_initialization(
            self,
            data_type=DataAlias.NODE_AUTHENTIC,
        )
        subscribers_list_initialization(
            self,
            data_type=DataAlias.CONNECTIONS_AUTHENTIC,
        )
        subscribers_list_initialization(
            self,
            data_type=DataAlias.CONNECTIONS_SYNTHETIC,
        )
        subscribers_list_initialization(
            self,
            data_type=DataAlias.CONNECTIONS_NULL,
        )

        create_caches_general(self)
        create_caches_specialized(self)


def create_storage():
    return StorageStruct()
