from typing import TYPE_CHECKING
from typing import Dict
from src.runtime_storages.cache_abstract import CacheAbstract
from typing import List

from src.runtime_storages.other import cache_general_get
from src.runtime_storages.types import NodeAuthenticData, CacheGeneralAlias

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


class CacheNodesIndexes(CacheAbstract):
    """
    Keeps track of th indexes of each node
    """

    def __init__(self):
        self.cache_map: Dict[str, any] = {}

    def read(self, node_name: str) -> int:
        return self.cache_map[node_name]


def on_create_nodes(storage: 'StorageStruct',
                    new_nodes: List[NodeAuthenticData]) -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_INDEX_MAP)
    if not isinstance(self, CacheNodesIndexes):
        raise ValueError(f"Expected CacheNodesIndexes, got {type(self)}")

    # calculating the number of nodes inside the index cache
    start_index = len(storage.nodes_authentic) - len(new_nodes)
    for i, new_node in enumerate(new_nodes):
        self.cache_map[new_node["name"]] = start_index + i


def on_update_nodes(storage: 'StorageStruct',
                    old_nodes: List[NodeAuthenticData], new_nodes: List[NodeAuthenticData]) -> None:
    pass


def on_delete_nodes(storage: 'StorageStruct',
                    deleted_nodes: List[NodeAuthenticData]) -> None:
    on_invalidate_and_recalculate(storage)


def on_invalidate_and_recalculate(storage: 'StorageStruct') -> None:
    self = cache_general_get(storage, CacheGeneralAlias.NODE_INDEX_MAP)
    self.cache_map = {}
    for i, node in enumerate(storage.nodes_authentic):
        self.cache_map[node["name"]] = i


def validate_cache_nodes_indexes(cache: CacheAbstract) -> CacheNodesIndexes:
    if not isinstance(cache, CacheNodesIndexes):
        raise ValueError(f"Expected CacheNodesIndexes, got {type(cache)}")
    return cache
