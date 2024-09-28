from src.navigation_core.pure_functions import check_connection_already_existing, build_connection_name
from src.runtime_storages.storage_struct import StorageStruct
from typing import List, Callable
from tqdm import tqdm
from src import runtime_storages as storage

from src.runtime_storages.types import NodeAuthenticData, ConnectionSyntheticData


def build_augmented_connections(storage_struct: StorageStruct,
                                metric: Callable[[StorageStruct, NodeAuthenticData], List[str]],
                                nodes_names: List[str]
                                ) -> List[ConnectionSyntheticData]:
    synthetics_connections: List[ConnectionSyntheticData] = []
    found_connections: List[str] = []

    for idx, current_name in enumerate(tqdm(nodes_names)):
        adjacent_node = storage.node_get_by_name(storage_struct, current_name)

        found_nodes: List[str] = metric(storage_struct, adjacent_node)
        found_connections.extend(found_nodes)

        valid_adjacent_nodes = []
        for adjacent_node in found_nodes:
            if check_connection_already_existing(synthetics_connections, current_name, adjacent_node):
                continue
            if check_connection_already_existing(storage.connections_all_get(storage_struct), current_name,
                                                 adjacent_node):
                continue
            valid_adjacent_nodes.append(adjacent_node)

        for adjacent_node in valid_adjacent_nodes:
            name = build_connection_name(current_name, adjacent_node)
            connection = ConnectionSyntheticData(
                name=name,
                start=current_name,
                end=adjacent_node,
                distance=None,
                direction=None
            )
            synthetics_connections.append(connection)

    return synthetics_connections
