from torch import nn
from typing import List
from src.runtime_storages.storage_struct import StorageStruct
from src import runtime_storages as storage
from src.runtime_storages.types import NodeAuthenticData, ConnectionNullData, ConnectionSyntheticData, \
    ConnectionAuthenticData
from src.save_load_handlers.data_handle import read_other_data_from_file


def load_storage_with_base_data(storage_struct: StorageStruct, nodes_filename: str,
                                connections_authentic_filename: str, connections_synthetic_filename: str,
                                connections_null_filename: str) -> None:
    nodes: List[NodeAuthenticData] = read_other_data_from_file(nodes_filename)
    connections_authentic = read_other_data_from_file(connections_authentic_filename)
    connections_synthetic = read_other_data_from_file(connections_synthetic_filename)
    connections_null = read_other_data_from_file(connections_null_filename)

    storage.crud.create_nodes(storage=storage_struct, nodes=nodes)
    storage.crud.create_connections_authentic(storage=storage_struct, new_connections=connections_authentic)
    storage.crud.create_connections_synthetic(storage=storage_struct, new_connections=connections_synthetic)
    storage.crud.create_connections_null(storage=storage_struct, new_connections=connections_null)
