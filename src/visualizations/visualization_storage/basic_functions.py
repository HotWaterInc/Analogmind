from typing import List, Dict, Tuple, TYPE_CHECKING

from src.runtime_storages.types import Coords
from src.utils.configs_loader import load_config_ini
from src.visualizations.visualization_storage.types import NodesMapping
from src import runtime_storages as storage

import numpy as np

if TYPE_CHECKING:
    from src.runtime_storages.storage_struct import StorageStruct
    from src.visualizations.visualization_storage.visualization_struct import VisualizationDataStruct


def set_nodes_coordinates_map(visualization_struct: 'VisualizationDataStruct',
                              nodes_coordinates_map: NodesMapping) -> None:
    visualization_struct.nodes_coordinates_map = nodes_coordinates_map


def get_nodes_coordinates_map(visualization_struct: 'VisualizationDataStruct') -> NodesMapping:
    return visualization_struct.nodes_coordinates_map


def build_nodes_coordinates_map(visualization_struct: 'VisualizationDataStruct', storage_struct: 'StorageStruct',
                                percent: float) -> None:
    nodes = storage.nodes_get_all_names(storage_struct)
    sampled_node_names = np.random.choice(nodes, int(len(nodes) * percent), replace=False)
    datapoints_coordinates_map: NodesMapping = {}

    for name in sampled_node_names:
        coords = storage.node_get_coords_metadata(storage_struct, name)
        x, y = coords
        datapoints_coordinates_map[name] = Coords(x=x, y=y)

    set_nodes_coordinates_map(visualization_struct, datapoints_coordinates_map)


def recenter_datapoints_coordinates_map(visualization_struct: 'VisualizationDataStruct') -> None:
    """
    Recenter the coordinates map so that the center of the coordinates is 0,0
    """
    nodes_coordinates_map = get_nodes_coordinates_map(visualization_struct)
    x_mean, y_mean = 0, 0
    total_datapoints = len(nodes_coordinates_map)

    for key in nodes_coordinates_map:
        x_mean += nodes_coordinates_map[key]["x"]
        y_mean += nodes_coordinates_map[key]["y"]

    x_mean /= total_datapoints
    y_mean /= total_datapoints

    for key in nodes_coordinates_map:
        nodes_coordinates_map[key]["x"] -= x_mean
        nodes_coordinates_map[key]["y"] -= y_mean
