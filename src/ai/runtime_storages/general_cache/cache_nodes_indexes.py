from typing import TYPE_CHECKING
from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.functions.cache_functions import cache_general_get
from src.ai.runtime_storages.types import NodeData, CacheGeneralAlias
from typing import Dict

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


class Cache(CacheAbstract):
    """
    Keeps track of th indexes of each node
    """

    def __init__(self):
        self.cache_map: Dict[str, any] = {}


def on_create_node(storage: 'StorageStruct',
                   new_node: NodeData) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_INDEX_MAP)
    self.cache_map[new_node["name"]] = len(storage.environment_nodes_authentic) - 1


def on_update_node(storage: 'StorageStruct',
                   old_node: NodeData, new_node: NodeData) -> None:
    pass


def on_delete_node(storage: 'StorageStruct',
                   deleted_node: NodeData) -> None:
    on_invalidate_and_recalculate(storage)


def on_invalidate_and_recalculate(storage: 'StorageStruct') -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_INDEX_MAP)
    self.cache_map = {}
    for i, node in enumerate(storage.environment_nodes_authentic):
        self.cache_map[node["name"]] = i
