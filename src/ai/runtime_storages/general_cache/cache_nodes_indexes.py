from typing import TYPE_CHECKING
from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.functions.cache_functions import cache_general_get
from src.ai.runtime_storages.types import NodeAuthenticData, CacheGeneralAlias
from typing import Dict

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


class CacheNodesIndexes(CacheAbstract):
    """
    Keeps track of th indexes of each node
    """

    def __init__(self):
        self.cache_map: Dict[str, any] = {}

    def read(self, node_name: str) -> int:
        return self.cache_map[node_name]


def on_create_node(storage: 'StorageStruct',
                   new_node: NodeAuthenticData) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_INDEX_MAP)
    self.cache_map[new_node["name"]] = len(storage.nodes_authentic) - 1


def on_update_node(storage: 'StorageStruct',
                   old_node: NodeAuthenticData, new_node: NodeAuthenticData) -> None:
    pass


def on_delete_node(storage: 'StorageStruct',
                   deleted_node: NodeAuthenticData) -> None:
    on_invalidate_and_recalculate(storage)


def on_invalidate_and_recalculate(storage: 'StorageStruct') -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_INDEX_MAP)
    self.cache_map = {}
    for i, node in enumerate(storage.nodes_authentic):
        self.cache_map[node["name"]] = i
