from src.ai.runtime_storages.new.functionalities.functionalities_types import FunctionalityAlias
from src.ai.runtime_storages.new.functions.cache_functions import cache_specialized_get
from src.ai.runtime_storages.storage_struct import StorageStruct


def get_all_connections(storage: StorageStruct):
    cache = cache_specialized_get(storage, FunctionalityAlias.GET_ALL_CONNECTIONS)
