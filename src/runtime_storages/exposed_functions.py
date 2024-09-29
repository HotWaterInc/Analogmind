from typing import List, TYPE_CHECKING
import random
import torch
from src.navigation_core.autonomous_exploration.params import IS_CLOSE_THRESHOLD
from src.navigation_core.pure_functions import calculate_coords_distance, connection_reverse_order
from src.runtime_storages.other.cache_functions import cache_general_get
from src.runtime_storages.functions.pure_functions import eulerian_distance
from src.runtime_storages.crud.crud_functions import update_nodes_by_index
from src.runtime_storages.general_cache.cache_nodes_map import validate_cache_nodes_map
from src.runtime_storages.general_cache.cache_nodes_indexes import \
    validate_cache_nodes_indexes
from src.runtime_storages.types import ConnectionAuthenticData, NodeAuthenticData, CacheGeneralAlias, \
    ConnectionNullData, ConnectionSyntheticData
from src.utils.utils import array_to_tensor

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct

from .functions.basic_functions import (
    connections_authentic_get,
    nodes_get_all_names,
    nodes_get_datapoints_arrays,
    connections_authentic_sample,
    node_get_by_name,
    node_get_by_index,
    node_get_datapoint_tensor_at_index,
    node_get_datapoints_by_name,
    node_get_index_by_name,
    connections_all_get,
    connections_authentic_get,
    connections_classify_into_authentic_synthetic,
    connection_null_get_all,
    node_get_datapoints_tensor,
    node_get_datapoints_count,
    node_get_coords_metadata,
    node_get_connections_adjacent,
    check_node_is_known_from_metadata,
    get_distance_between_nodes_metadata,
    get_direction_between_nodes_metadata,
    connections_authentic_check_if_exists,
    connections_synthetic_check_if_exists,
    connections_synthetic_get,
    node_get_connections_null,
    nodes_get_all
)
