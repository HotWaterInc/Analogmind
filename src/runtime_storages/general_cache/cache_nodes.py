from typing import TYPE_CHECKING
from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.functions.cache_functions import cache_general_get
from src.ai.runtime_storages.types import NodeAuthenticData, CacheGeneralAlias

from typing import Dict

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


class CacheNodes(CacheAbstract):
    """
    Fast retrieval for nodes from a map
    """

    def __init__(self):
        self.cache_map: Dict[str, any] = {}

    def invalidate_and_recalculate(self, storage):
        pass


def on_create_node(storage: 'StorageStruct',
                   new_node: NodeAuthenticData) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    self.cache_map[new_node["name"]] = new_node


def on_update_node(storage: 'StorageStruct',
                   old_node: NodeAuthenticData, new_node: NodeAuthenticData) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    self.cache_map[new_node["name"]] = new_node


def on_delete_node(storage: 'StorageStruct',
                   deleted_node: NodeAuthenticData) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    del self.cache_map[deleted_node["name"]]
