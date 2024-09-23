from typing import TYPE_CHECKING
from src.runtime_storages.types import DataAlias, OperationsAlias

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


def subscribers_list_initialization(storage: 'StorageStruct', data_type: DataAlias):
    storage.data_crud_subscribers[data_type] = {
        OperationsAlias.CREATE: [],
        OperationsAlias.UPDATE: [],
        OperationsAlias.DELETE: [],
        OperationsAlias.READ: [],
    }


def subscribe_to_crud_operations(storage: 'StorageStruct', data_alias: DataAlias, create_subscriber,
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


def subscribe_to_crud_operation(storage: 'StorageStruct', data_alias: DataAlias, operation_type: OperationsAlias,
                                subscriber):
    storage.data_crud_subscribers[data_alias][operation_type].append(subscriber)
