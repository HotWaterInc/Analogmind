from functools import wraps

from src.runtime_storages.types import DataAlias, OperationsAlias


def trigger_create_subscribers(data_alias: DataAlias):
    operation_alias = OperationsAlias.CREATE

    def function_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            storage = kwargs.get('storage', args[0] if args else None)
            element = func(*args, **kwargs)
            subscribers = storage.data_crud_subscribers[data_alias][operation_alias]
            for subscriber in subscribers:
                subscriber(storage, element)

        return wrapper

    return function_decorator


def trigger_update_subscribers(data_alias: DataAlias):
    operation_alias = OperationsAlias.UPDATE

    def function_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            storage = args[0]
            old, new = func(*args, **kwargs)
            subscribers = storage.data_crud_subscribers[data_alias][operation_alias]
            for subscriber in subscribers:
                subscriber(storage, old, new)

        return wrapper

    return function_decorator


def trigger_delete_subscribers(data_alias: DataAlias):
    operation_alias = OperationsAlias.DELETE

    def function_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            storage = args[0]
            deleted_element = func(*args, **kwargs)
            subscribers = storage.data_crud_subscribers[data_alias][operation_alias]
            for subscriber in subscribers:
                subscriber(storage, deleted_element)

        return wrapper

    return function_decorator
