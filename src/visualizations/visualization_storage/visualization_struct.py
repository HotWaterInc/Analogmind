from dataclasses import dataclass, field
from typing import List, Dict
from .types import NodesMapping


@dataclass
class VisualizationDataStruct:
    nodes_coordinates_map: NodesMapping = field(default_factory=dict)
    params: Dict = field(default_factory=dict)

    def __post_init__(self):
        pass
