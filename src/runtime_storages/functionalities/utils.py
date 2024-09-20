from typing import TYPE_CHECKING
from src.ai.runtime_storages.functionalities import get_all_connections
from src.ai.runtime_storages.functionalities.functionalities_types import FunctionalityAlias

if TYPE_CHECKING:
    from src.ai.runtime_storages.storage_struct import StorageStruct


def cache_functionalities_create_new(storage: 'StorageStruct', functionality_type: FunctionalityAlias):
    if functionality_type == FunctionalityAlias.GET_ALL_CONNECTIONS:
        storage.caches_functionalities[functionality_type] = get_all_connections.CacheNodesIndexes()
