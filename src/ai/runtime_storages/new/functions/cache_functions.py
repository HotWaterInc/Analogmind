from typing import TYPE_CHECKING
from src.ai.runtime_storages.new.cache.cache_abstract import CacheAbstract
from src.ai.runtime_storages.new.cache.cache_nodes_class import CacheMapNodes
from src.ai.runtime_storages.new.functions.subscriber_functions import \
    subscribe_to_crud_operations
from src.ai.runtime_storages.new.types import NodeData, ConnectionData, DataAlias, OperationsAlias

if TYPE_CHECKING:
    from src.ai.runtime_storages.new.storage_struct import StorageStruct


def create_baseline_caches(storage: 'StorageStruct'):
    """
    Creates caches which are simple and relied upon by many functions
    """
    print("create caches")
    cache_map_create_new(storage, storage.NODE_CACHE_MAP_ID, CacheMapNodes())
    subscribe_to_crud_operations(
        storage=storage,
        data_alias=DataAlias.NODE_DATA,
        create_subscriber=storage.caches[storage.NODE_CACHE_MAP_ID].create,
        update_subscriber=storage.caches[storage.NODE_CACHE_MAP_ID].update,
        delete_subscriber=storage.caches[storage.NODE_CACHE_MAP_ID].delete
    )


def cache_map_create_new(storage: 'StorageStruct', name: str, cache: CacheAbstract):
    storage.caches[name] = cache
