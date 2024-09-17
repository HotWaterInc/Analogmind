from typing import TYPE_CHECKING

from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.new.functions.cache_functions import cache_general_get
from src.ai.runtime_storages.new.types import NodeData, CachesGeneralAlias
from typing import Dict

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


class Cache(CacheAbstract):
    def __init__(self):
        self.cache_map: Dict[str, any] = {}

    def invalidate_and_recalculate(self, storage):
        pass


def on_create_node(storage: 'StorageStruct',
                   new_node: NodeData) -> None:
    self = cache_general_get(storage, CachesGeneralAlias.NODE_CACHE_MAP_ID)
    self.cache_map[new_node["name"]] = new_node


def on_update_node(storage: 'StorageStruct',
                   new_node: NodeData) -> None:
    self = cache_general_get(storage, CachesGeneralAlias.NODE_CACHE_MAP_ID)
    self.cache_map[new_node["name"]] = new_node


def on_delete_node(storage: 'StorageStruct',
                   node_name: str) -> None:
    self = cache_general_get(storage, CachesGeneralAlias.NODE_CACHE_MAP_ID)
    del self.cache_map[node_name]
