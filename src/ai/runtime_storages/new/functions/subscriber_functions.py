from typing import TYPE_CHECKING

from src.ai.runtime_storages.new.types import NodeData, ConnectionData, DataAlias, OperationsAlias

if TYPE_CHECKING:
    from src.ai.runtime_storages.new.storage_struct import StorageStruct


def subscribers_list_initialization(self, data_type: DataAlias):
    self.DATA_CRUD_SUBSCRIBERS[data_type] = {
        OperationsAlias.CREATE: [],
        OperationsAlias.UPDATE: [],
        OperationsAlias.DELETE: [],
        OperationsAlias.READ: [],
    }


def subscribe_to_crud_operations(storage, data_alias: DataAlias, create_subscriber,
                                 update_subscriber, delete_subscriber) -> None:
    subscribe_to_crud_operation(
        storage=storage,
        data_alias=data_alias,
        operation_type=OperationsAlias.CREATE,
        subscriber=create_subscriber
    )
    subscribe_to_crud_operation(
        storage=storage,
        data_alias=data_alias,
        operation_type=OperationsAlias.UPDATE,
        subscriber=update_subscriber
    )
    subscribe_to_crud_operation(
        storage=storage,
        data_alias=data_alias,
        operation_type=OperationsAlias.DELETE,
        subscriber=delete_subscriber
    )


def subscribe_to_crud_operation(storage, data_alias: DataAlias, operation_type: OperationsAlias, subscriber):
    storage.DATA_CRUD_SUBSCRIBERS[data_alias][operation_type].append(subscriber)
