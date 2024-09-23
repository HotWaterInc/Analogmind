from dataclasses import dataclass, field
from typing import List, Dict
from src.runtime_storages.cache_abstract import CacheAbstract
from src.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.runtime_storages.functions.subscriber_functions import subscribers_list_initialization
from src.runtime_storages.other import create_caches_specialized, create_caches_general
from src.runtime_storages.types import DataAlias

from src.runtime_storages.types import NodeAuthenticData, ConnectionNullData, ConnectionSyntheticData, \
    ConnectionAuthenticData, CacheGeneralAlias


@dataclass
class StorageStruct:
    nodes_authentic: List[NodeAuthenticData] = field(default_factory=list)

    connections_authentic: List[ConnectionAuthenticData] = field(default_factory=list)
    connections_synthetic: List[ConnectionSyntheticData] = field(default_factory=list)
    connections_null: List[ConnectionNullData] = field(default_factory=list)

    caches: Dict[CacheGeneralAlias, CacheAbstract] = field(default_factory=dict)
    caches_functionalities: Dict[FunctionalityAlias, CacheAbstract] = field(default_factory=dict)
    data_crud_subscribers: Dict[DataAlias, any] = field(default_factory=dict)

    transformation: any = None

    def __post_init__(self):
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
