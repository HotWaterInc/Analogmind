import random
from typing import TYPE_CHECKING

from src.ai.runtime_storages.functions.basic_functions import node_get_coords_metadata

if TYPE_CHECKING:
    from src.ai.runtime_storages.visualization_data_struct import VisualizationDataStruct
    from src.ai.runtime_storages.storage_struct import StorageStruct


def recenter_nodes_coordinates(visualization_struct: VisualizationDataStruct):
    nodes_coordinates_map = visualization_struct.nodes_coordinates
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


def build_nodes_coordinates(storage_struct: 'StorageStruct', visualization_struct: VisualizationDataStruct):
    nodes = storage_struct.nodes_authentic
    for node in nodes:
        name = node["name"]
        coords = node_get_coords_metadata(storage_struct, name)
        x, y = coords.x, coords.y
        visualization_struct.nodes_coordinates[name] = _create_node_coordinates(x, y)


def build_nodes_coordinates_sparse(storage_struct: 'StorageStruct', visualization_struct: VisualizationDataStruct,
                                   percent: float = 0.25):
    nodes = storage_struct.nodes_authentic
    selected_nodes = random.sample(nodes, int(len(nodes) * percent))
    for node in selected_nodes:
        name = node["name"]
        coords = node_get_coords_metadata(storage_struct, name)
        x, y = coords.x, coords.y
        visualization_struct.nodes_coordinates[name] = _create_node_coordinates(x, y)


def _create_node_coordinates(x, y):
    return {"x": x, "y": y}
