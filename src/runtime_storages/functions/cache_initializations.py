from typing import TYPE_CHECKING
from src.runtime_storages.functionalities.utils import cache_functionalities_create_new
from src.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.runtime_storages.functions.cache_functions import cache_registration
from src.runtime_storages.functions.subscriber_functions import \
    subscribe_to_crud_operations
from src.runtime_storages.types import DataAlias, CacheGeneralAlias
from src.runtime_storages.general_cache import cache_nodes_map, cache_nodes_indexes
from src.runtime_storages.functionalities import get_walk_distance

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


def create_caches_specialized(storage: 'StorageStruct'):
    cache_functionalities_create_new(storage, FunctionalityAlias.GET_WALK_DISTANCE)
    subscribe_to_crud_operations(
        storage=storage,
        data_alias=DataAlias.CONNECTIONS_AUTHENTIC,
        create_subscriber=get_walk_distance.on_create_connections,
        update_subscriber=get_walk_distance.on_update_connections,
        delete_subscriber=get_walk_distance.on_delete_connections
    )
    subscribe_to_crud_operations(
        storage=storage,
        data_alias=DataAlias.CONNECTIONS_SYNTHETIC,
        create_subscriber=get_walk_distance.on_create_connections,
        update_subscriber=get_walk_distance.on_update_connections,
        delete_subscriber=get_walk_distance.on_delete_connections
    )
    subscribe_to_crud_operations(
        storage=storage,
        data_alias=DataAlias.NODE_AUTHENTIC,
        create_subscriber=get_walk_distance.on_create_nodes,
        update_subscriber=get_walk_distance.on_update_nodes,
        delete_subscriber=get_walk_distance.on_delete_nodes
    )


def create_caches_general(storage: 'StorageStruct'):
    """
    Creates caches which are simple and relied upon by many functions
    """
    cache_registration(storage, CacheGeneralAlias.NODE_CACHE_MAP, cache_nodes_map.CacheNodesMap())
    subscribe_to_crud_operations(
        storage=storage,
        data_alias=DataAlias.NODE_AUTHENTIC,
        create_subscriber=cache_nodes_map.on_create_nodes,
        update_subscriber=cache_nodes_map.on_update_nodes,
        delete_subscriber=cache_nodes_map.on_delete_nodes
    )

    cache_registration(storage, CacheGeneralAlias.NODE_INDEX_MAP, cache_nodes_indexes.CacheNodesAuthenticIndexes())
    subscribe_to_crud_operations(
        storage=storage,
        data_alias=DataAlias.NODE_AUTHENTIC,
        create_subscriber=cache_nodes_indexes.on_create_nodes,
        update_subscriber=cache_nodes_indexes.on_update_nodes,
        delete_subscriber=cache_nodes_indexes.on_delete_nodes
    )
