from typing import List, TYPE_CHECKING
import copy
from src.runtime_storages.types import NodeAuthenticData, ConnectionAuthenticData, ConnectionSyntheticData, \
    ConnectionNullData
from src.runtime_storages.functions.method_decorators import trigger_update_subscribers, \
    trigger_create_subscribers, \
    trigger_delete_subscribers
from src.runtime_storages.types import DataAlias

if TYPE_CHECKING:
    from src.runtime_storages import StorageStruct


@trigger_create_subscribers(data_alias=DataAlias.NODE_AUTHENTIC,
                            )
def create_nodes(storage: 'StorageStruct', nodes: List[NodeAuthenticData]) -> List[NodeAuthenticData]:
    storage.nodes_authentic.extend(nodes)
    return nodes


@trigger_delete_subscribers(data_alias=DataAlias.NODE_AUTHENTIC,
                            )
def delete_nodes(storage: 'StorageStruct', names: List[str]) -> List[NodeAuthenticData]:
    target_indexes = []
    for i, node in enumerate(storage.nodes_authentic):
        if node["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Node with name {names} not found.")

    deleted_nodes = [storage.nodes_authentic[i] for i in target_indexes]
    for i in target_indexes:
        del storage.nodes_authentic[i]

    return deleted_nodes


@trigger_update_subscribers(data_alias=DataAlias.NODE_AUTHENTIC,
                            )
def update_nodes_by_index(storage: 'StorageStruct', indexes: List[int], updated_nodes: List[NodeAuthenticData]) -> \
        tuple[
            List[any], List[any]]:
    current_nodes = [storage.nodes_authentic[i] for i in indexes]
    current_nodes_copy = copy.deepcopy(current_nodes)

    for i, index in enumerate(indexes):
        new_node = updated_nodes[i]
        old_node = current_nodes[i]
        if new_node["name"] is not None:
            old_node["name"] = new_node["name"]
        if new_node["datapoints_array"] is not None:
            old_node["datapoints_array"] = new_node["datapoints_array"]
        if new_node["params"] is not None:
            old_node["params"] = new_node["params"]

    return current_nodes_copy, current_nodes


@trigger_update_subscribers(data_alias=DataAlias.NODE_AUTHENTIC,
                            )
def update_nodes_by_name(storage: 'StorageStruct', names: List[str], updated_nodes: List[NodeAuthenticData]) -> tuple[
    List[any], List[any]]:
    target_indexes = []
    for i, node in enumerate(storage.nodes_authentic):
        if node["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Node with name {names} not found.")

    current_nodes = [storage.nodes_authentic[i] for i in target_indexes]
    current_nodes_copy = copy.deepcopy(current_nodes)

    for i, index in enumerate(target_indexes):
        new_node = updated_nodes[i]
        old_node = current_nodes[i]
        if new_node["name"] is not None:
            old_node["name"] = new_node["name"]
        if new_node["datapoints_array"] is not None:
            old_node["datapoints_array"] = new_node["datapoints_array"]
        if new_node["params"] is not None:
            old_node["params"] = new_node["params"]

    return current_nodes_copy, current_nodes


@trigger_create_subscribers(data_alias=DataAlias.CONNECTIONS_AUTHENTIC,
                            )
def create_connections_authentic(storage: 'StorageStruct', new_connections: List[ConnectionAuthenticData]) -> List[
    ConnectionAuthenticData]:
    storage.connections_authentic.extend(new_connections)
    return new_connections


@trigger_delete_subscribers(data_alias=DataAlias.CONNECTIONS_AUTHENTIC,
                            )
def delete_connections_authentic(storage, names: str) -> List[ConnectionAuthenticData]:
    target_indexes = []
    for i, connection in enumerate(storage.connections_authentic):
        if connection["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Not all connections found to delete.")

    deleted_connections = [storage.connections_authentic[i] for i in target_indexes]
    for i in target_indexes:
        del storage.connections_authentic[i]

    return deleted_connections


@trigger_update_subscribers(data_alias=DataAlias.CONNECTIONS_AUTHENTIC,
                            )
def update_connections_authentic(storage: 'StorageStruct', names: List[str],
                                 updated_connections: List[ConnectionAuthenticData]) -> tuple[List[any], List[any]]:
    target_indexes = []
    for i, connection in enumerate(storage.connections_authentic):
        if connection["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Not all connections found to update.")

    current_connections = [storage.connections_authentic[i] for i in target_indexes]
    current_connections_copy = copy.deepcopy(current_connections)

    for i, index in enumerate(target_indexes):
        new_connection = updated_connections[i]
        old_connection = current_connections[i]
        if new_connection["name"] is not None:
            old_connection["name"] = new_connection["name"]
        if new_connection["start"] is not None:
            old_connection["start"] = new_connection["start"]
        if new_connection["end"] is not None:
            old_connection["end"] = new_connection["end"]
        if new_connection["distance"] is not None:
            old_connection["distance"] = new_connection["distance"]
        if new_connection["direction"] is not None:
            old_connection["direction"] = new_connection["direction"]

    return current_connections_copy, current_connections


@trigger_create_subscribers(data_alias=DataAlias.CONNECTIONS_SYNTHETIC)
def create_connections_synthetic(storage, new_connections: List[ConnectionSyntheticData]) -> List[
    ConnectionSyntheticData]:
    storage.connections_synthetic.extend(new_connections)
    return new_connections


@trigger_delete_subscribers(data_alias=DataAlias.CONNECTIONS_SYNTHETIC)
def delete_connections_synthetic(storage, names: List[str]) -> List[ConnectionSyntheticData]:
    target_indexes = []
    for i, connection in enumerate(storage.connections_synthetic):
        if connection["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Not all connections found to delete.")

    deleted_connections = [storage.connections_synthetic[i] for i in target_indexes]
    for i in target_indexes:
        del storage.connections_synthetic[i]

    return deleted_connections


@trigger_update_subscribers(data_alias=DataAlias.CONNECTIONS_SYNTHETIC)
def update_connections_synthetic(storage, names: List[str], updated_connections: List[ConnectionSyntheticData]) -> \
        tuple[
            List[any], List[any]]:
    target_indexes = []
    for i, connection in enumerate(storage.connections_synthetic):
        if connection["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Not all connections found to update.")

    current_connections = [storage.connections_synthetic[i] for i in target_indexes]
    current_connections_copy = copy.deepcopy(current_connections)

    for i, index in enumerate(target_indexes):
        new_connection = updated_connections[i]
        old_connection = current_connections[i]
        if new_connection["name"] is not None:
            old_connection["name"] = new_connection["name"]
        if new_connection["start"] is not None:
            old_connection["start"] = new_connection["start"]
        if new_connection["end"] is not None:
            old_connection["end"] = new_connection["end"]
        if new_connection["distance"] is not None:
            old_connection["distance"] = new_connection["distance"]
        if new_connection["direction"] is not None:
            old_connection["direction"] = new_connection["direction"]

    return current_connections_copy, current_connections


@trigger_create_subscribers(data_alias=DataAlias.CONNECTIONS_NULL)
def create_connections_null(storage, connections: List[ConnectionNullData]) -> List[ConnectionNullData]:
    storage.connections_null.extend(connections)
    return connections


@trigger_delete_subscribers(data_alias=DataAlias.CONNECTIONS_NULL)
def delete_connections_null(storage, names: List[str]) -> List[ConnectionAuthenticData]:
    target_indexes = []
    for i, connection in enumerate(storage.connections_null):
        if connection["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Not all connections found to delete.")

    deleted_connections = [storage.connections_null[i] for i in target_indexes]
    for i in target_indexes:
        del storage.connections_null[i]

    return deleted_connections


@trigger_update_subscribers(data_alias=DataAlias.CONNECTIONS_NULL)
def update_connections_null(storage, names: List[str], updated_connections: List[ConnectionAuthenticData]) -> tuple[
    List[any], List[any]]:
    target_indexes = []
    for i, connection in enumerate(storage.connections_null):
        if connection["name"] in names:
            target_indexes.append(i)

    if len(target_indexes) != len(names):
        raise ValueError(f"Not all connections found to update.")

    current_connections = [storage.connections_null[i] for i in target_indexes]
    current_connections_copy = copy.deepcopy(current_connections)

    for i, index in enumerate(target_indexes):
        new_connection = updated_connections[i]
        old_connection = current_connections[i]
        if new_connection["name"] is not None:
            old_connection["name"] = new_connection["name"]
        if new_connection["start"] is not None:
            old_connection["start"] = new_connection["start"]
        if new_connection["end"] is not None:
            old_connection["end"] = new_connection["end"]
        if new_connection["distance"] is not None:
            old_connection["distance"] = new_connection["distance"]
        if new_connection["direction"] is not None:
            old_connection["direction"] = new_connection["direction"]

    return current_connections_copy, current_connections
