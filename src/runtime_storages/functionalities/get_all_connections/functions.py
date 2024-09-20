from src.ai.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.ai.runtime_storages.functionalities.get_all_connections import CacheGetAllConnections
from src.ai.runtime_storages.functions.cache_functions import cache_specialized_get
from src.ai.runtime_storages.storage_struct import StorageStruct


def get_all_connections(storage: StorageStruct):
    cache = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
    if not isinstance(cache, CacheGetAllConnections):
        raise ValueError(f"Expected CacheGetAllConnections, got {type(cache)}")
    
    return cache.connections
