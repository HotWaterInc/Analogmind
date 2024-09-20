from typing import TYPE_CHECKING
from typing import List
from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.ai.runtime_storages.functions.cache_functions import cache_specialized_get
from src.ai.runtime_storages.types import ConnectionNullData, ConnectionAuthenticData, ConnectionSyntheticData

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


class CacheGetAllConnections(CacheAbstract):
    def __init__(self):
        self.connections: List[ConnectionAuthenticData | ConnectionSyntheticData] = []

    def invalidate_and_recalculate(self, storage):
        pass


def on_create_connection(storage: 'StorageStruct',
                         new_connection) -> None:
    self = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
    if not isinstance(self, CacheGetAllConnections):
        raise ValueError(f"Expected CacheGetAllConnections, got {type(self)}")
    self.connections.append(new_connection)


def on_update_connection(storage: 'StorageStruct',
                         new_connection: ConnectionAuthenticData | ConnectionSyntheticData) -> None:
    self = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
    connections: List[ConnectionAuthenticData | ConnectionSyntheticData] = self.connections
    target_index = None

    for i, connection in enumerate(connections):
        if connection["name"] == new_connection["name"]:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection {new_connection['name']} not found")

    self.connections[target_index] = new_connection


def on_delete_connection(storage: 'StorageStruct',
                         deleted_connection: ConnectionAuthenticData | ConnectionSyntheticData) -> None:
    self = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
    connections: List[ConnectionAuthenticData | ConnectionSyntheticData] = self.connections
    target_index = None

    for i, connection in enumerate(connections):
        if connection["name"] == deleted_connection["name"]:
            target_index = i
            break

    if target_index is None:
        raise ValueError(f"Connection {deleted_connection['name']} not found")

    self.connections.pop(target_index)
