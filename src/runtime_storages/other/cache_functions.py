from typing import TYPE_CHECKING

from src.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.runtime_storages.types import CacheGeneralAlias
from src.runtime_storages.cache_abstract import CacheAbstract

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct


def cache_registration(storage: 'StorageStruct', alias: 'CacheGeneralAlias', cache: 'CacheAbstract'):
    storage.caches[alias] = cache


def cache_general_get(storage: 'StorageStruct', cache_type: 'CacheGeneralAlias') -> 'CacheAbstract':
    return storage.caches[cache_type]


def cache_specialized_get(storage: 'StorageStruct', cache_type: FunctionalityAlias):
    return storage.caches_functionalities[cache_type]
