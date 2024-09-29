from dataclasses import dataclass, field
from typing import List, Dict
from .types import NodesMapping, MobjectsParams


@dataclass
class VisualizationDataStruct:
    nodes_coordinates_map: NodesMapping = field(default_factory=dict)
    params: MobjectsParams = field(default_factory=dict)

    def __post_init__(self):
        pass


def create_visualization_struct():
    return VisualizationDataStruct()
