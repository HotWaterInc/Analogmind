from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class VisualizationDataStruct:
    nodes_coordinates: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        pass
