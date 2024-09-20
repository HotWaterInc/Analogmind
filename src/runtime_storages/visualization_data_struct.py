from dataclasses import dataclass, field
from typing import List, Dict
from src.ai.runtime_storages.cache_abstract import CacheAbstract
from src.ai.runtime_storages.functionalities.functionalities_types import FunctionalityAlias
from src.ai.runtime_storages.functions.cache_functions import create_caches_general, create_caches_specialized
from src.ai.runtime_storages.functions.subscriber_functions import subscribers_list_initialization
from src.ai.runtime_storages.types import NodeAuthenticData, ConnectionAuthenticData, DataAlias, CacheGeneralAlias, \
    ConnectionSyntheticData, ConnectionNullData


@dataclass
class VisualizationDataStruct:
    nodes_coordinates: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        pass
