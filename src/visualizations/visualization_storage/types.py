from typing import Dict, List, TypedDict

from src.runtime_storages.types import Coords

NodesMapping = Dict[str, Coords]


class MobjectsParams(TypedDict):
    radius: float
    distance_scale: float
