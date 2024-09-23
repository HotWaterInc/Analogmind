from src.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.runtime_storages.functionalities.get_walk_distance import CacheGetWalkDistance
from src.runtime_storages.other import cache_specialized_get
from src.runtime_storages.storage_struct import StorageStruct


def get_walk_distance(storage: StorageStruct, start_node: str, end_node: str):
    cache: CacheGetWalkDistance = cache_specialized_get(storage, FunctionalityAlias.GET_WALK_DISTANCE)
    if not isinstance(cache, CacheGetWalkDistance):
        raise ValueError("Cache is not of the correct type.")

    return cache.read(
        start_node=start_node,
        end_node=end_node,
    )
