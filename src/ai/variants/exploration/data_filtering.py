from src.ai.variants.exploration.algorithms import build_connections_hashmap, \
    find_minimum_distance_between_datapoints_on_graph_bfs
from src.ai.variants.exploration.exploration_evaluations import evaluate_distance_metric
from src.ai.runtime_data_storage.storage_superset2 import *
from src.ai.variants.exploration.metric_builders import build_find_adjacency_heursitic_adjacency_network, \
    build_find_adjacency_heursitic_raw_data
from src.ai.variants.exploration.params import REDUNDANCY_CONNECTION_ANGLE
from src.ai.variants.exploration.utils import check_datapoint_connections_completeness
from src.ai.variants.exploration.utils_pure_functions import direction_to_degrees_atan
from src.utils import get_device


def filtering_metric_djakstra(storage: StorageSuperset2, datapoint: str) -> any:
    """
    Filtering datapoints by checking if they add any additional pathways between any of their neighbors in the topological graph
    """
    connection_hashmap_without = build_connections_hashmap(storage.get_all_connections_only_datapoints(), [datapoint])

    connections_neighbors = storage.get_datapoint_adjacent_connections_non_null(datapoint)
    neighbors_count = len(connections_neighbors)

    is_redundant = True

    for idx in range(neighbors_count):
        for jdx in range(idx + 1, neighbors_count):
            start_neighbor = connections_neighbors[idx]["end"]
            end_neighbor = connections_neighbors[jdx]["end"]

            distance_to_start = connections_neighbors[idx]["distance"]
            distance_to_end = connections_neighbors[jdx]["distance"]

            distance_without_datapoint = find_minimum_distance_between_datapoints_on_graph_djakstra(start_neighbor,
                                                                                                    end_neighbor,
                                                                                                    connection_hashmap_without)

            if distance_without_datapoint >= (distance_to_start + distance_to_end) * 1.2:
                # not redundant if it bridges with better distance
                is_redundant = False

    return is_redundant


def filtering_redundancy_djakstra_based(storage: StorageSuperset2, datapoints: List[str]):
    count_redundant = 0
    count_not_redundant = 0
    total = 0

    for idx, datapoint in enumerate(datapoints):
        is_connection_complete = check_datapoint_connections_completeness(storage, datapoint)
        is_redundant = False
        if not is_connection_complete:
            print("Datapoint", datapoint, "is not connected completely")
        else:
            is_redundant = filtering_metric_djakstra(storage, datapoint)
        total += 1

        print("Processing", idx, "out of", len(datapoints))

        if is_redundant:
            count_redundant += 1
            storage.remove_datapoint(datapoint)
            print("Removing redundant datapoint")
        else:
            count_not_redundant += 1

    print("Count not redundant", count_not_redundant)
    print("Count redundant", count_redundant)
    print("Total", total)


def filtering_redundancy_neighbors_based(storage: StorageSuperset2, datapoints: List[str]):
    pass


def data_filtering_redundancies(storage: StorageSuperset2):
    """
    Filter redundant datapoints which don't add any value to the topological graph
    or enhance the neural networks in any way
    """

    datapoints = storage.get_all_datapoints()
    # filtering_redundancy_djakstra_based(storage, datapoints)
    # filtering_redundancy_neighbors_based()


def data_filtering_redundant_datapoints(storage: StorageSuperset2):
    datapoints = storage.get_all_datapoints()
    filtering_redundancy_djakstra_based(storage, datapoints)
    # filtering_redundancy_neighbors_based()
    pass


def data_filtering_redundant_connections(storage: StorageSuperset2):
    datapoints = storage.get_all_datapoints()
    count_redundant = 0
    total_count = 0

    for datapoint in datapoints:
        connections = storage.get_datapoint_adjacent_connections_direction_filled(datapoint)
        connections_count = len(connections)
        to_remove = []
        for idx in range(connections_count):
            for jdx in range(idx + 1, connections_count):
                first_connection = connections[idx]
                second_connection = connections[jdx]

                first_direction = first_connection["direction"]
                second_direction = second_connection["direction"]
                first_distance = first_connection["distance"]
                second_distance = second_connection["distance"]

                first_angle = direction_to_degrees_atan(first_direction)
                second_angle = direction_to_degrees_atan(second_direction)
                total_count += 1
                if abs(first_angle - second_angle) < REDUNDANCY_CONNECTION_ANGLE:
                    # print("Looks like redundant")
                    # print("Angles", first_angle, second_angle)
                    # invalidate the bigger distance
                    if first_distance > second_distance:
                        to_remove.append(first_connection)
                    else:
                        to_remove.append(second_connection)

        for connection in to_remove:
            count_redundant += storage.remove_connection(datapoint, connection["end"])

    print("Count redundant", count_redundant)
    print("Total count", total_count)
