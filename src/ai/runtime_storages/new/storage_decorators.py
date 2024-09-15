from src.ai.runtime_storages.new.types import NodeData, ConnectionData, TypeAlias, OperationsAlias
from functools import wraps


def crud_operation(data_alias: TypeAlias, operation_alias: OperationsAlias):
    def function_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            storage = args[0]

            result = func(*args, **kwargs)
            subscribers = storage.DATA_CRUD_SUBSCRIBERS[data_alias][operation_alias]
            for subscriber in subscribers:
                subscriber(result)

            return result

        return wrapper

    return function_decorator
