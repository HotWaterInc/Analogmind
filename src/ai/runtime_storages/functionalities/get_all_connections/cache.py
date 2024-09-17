from typing import TYPE_CHECKING
from typing import List

from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.new.functionalities.functionalities_types import FunctionalityAlias
from src.ai.runtime_storages.new.functions.cache_functions import cache_specialized_get
from src.ai.runtime_storages.new.types import ConnectionData

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


class Cache(CacheAbstract):
    def __init__(self):
        self.connections: List[ConnectionData] = []

    def read(self, storage):
        return self.connections

    def invalidate_and_recalculate(self, storage):
        pass


def on_create_connection(storage: 'StorageStruct',
                         new_connection) -> None:
    self = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
    self.connections.append(new_connection)


def on_update_connection(storage: 'StorageStruct',
                         new_connection: ConnectionData) -> None:
    self = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
    connections: List[ConnectionData] = self.connections
    target_index = None

    for i, connection in enumerate(connections):
        if connection["name"] == new_connection["name"]:
            target_index = i
            break

    self.connections[target_index] = new_connection


def on_delete_connection(storage: 'StorageStruct',
                         deleted_connection: ConnectionData) -> None:
    self = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
    connections: List[ConnectionData] = self.connections
    target_index = None

    for i, connection in enumerate(connections):
        if connection["name"] == deleted_connection["name"]:
            target_index = i
            break

    self.connections.pop(target_index)
