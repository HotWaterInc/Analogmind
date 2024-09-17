from src.ai.runtime_storages.new.functions.subscriber_functions import subscribers_list_initialization
from src.ai.runtime_storages.new.types import NodeData, ConnectionData, DataAlias, OperationsAlias
from functools import wraps


def trigger_crud_subscribers(data_alias: DataAlias, operation_alias: OperationsAlias):
    def function_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            storage = args[0]
            element = func(*args, **kwargs)
            subscribers = storage.data_crud_subscribers[data_alias][operation_alias]
            for subscriber in subscribers:
                subscriber(storage, element)

            return element

        return wrapper

    return function_decorator
