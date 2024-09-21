from typing import TYPE_CHECKING
from typing import List
from src.runtime_storages import NodeAuthenticData, cache_general_get, CacheGeneralAlias
from src.runtime_storages.cache_abstract import CacheAbstract

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


def on_create_nodes(storage: 'StorageStruct',
                    new_nodes: List[NodeAuthenticData]) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    if not isinstance(self, CacheNodes):
        raise ValueError("Cache is not of the correct type")

    for new_node in new_nodes:
        self.cache_map[new_node["name"]] = new_node


def on_update_nodes(storage: 'StorageStruct',
                    old_nodes: List[NodeAuthenticData], new_nodes: List[NodeAuthenticData]) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    if not isinstance(self, CacheNodes):
        raise ValueError("Cache is not of the correct type")

    for old_node, new_node in zip(old_nodes, new_nodes):
        del self.cache_map[old_node["name"]]
        self.cache_map[new_node["name"]] = new_node


def on_delete_nodes(storage: 'StorageStruct',
                    deleted_nodes: List[NodeAuthenticData]) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_CACHE_MAP)
    if not isinstance(self, CacheNodes):
        raise ValueError("Cache is not of the correct type")

    for deleted_node in deleted_nodes:
        del self.cache_map[deleted_node["name"]]
