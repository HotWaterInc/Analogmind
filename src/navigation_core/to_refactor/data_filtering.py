from src.navigation_core import build_connections_hashmap, \
    find_minimum_distance_between_datapoints_on_graph_bfs
from src.ai.runtime_storages.storage_superset2 import *
from src.navigation_core import REDUNDANCY_CONNECTION_ANGLE
from src.navigation_core import check_datapoint_connections_completeness, check_datapoint_density
from src.navigation_core import direction_to_degrees_atan


def filtering_metric_djakstra(storage: StorageSuperset2, datapoint: str) -> any:
    """
    Filtering datapoints by checking if they add any additional pathways between any of their neighbors in the topological graph
    """
    connection_hashmap_without = build_connections_hashmap(storage.get_all_connections_only_datapoints(), [datapoint])

    connections_neighbors = storage.connections_all_get(datapoint)
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

            if distance_without_datapoint >= (distance_to_start + distance_to_end) * 1.1:
                # not redundant if it bridges with better distance
                is_redundant = False

    return is_redundant


def filtering_redundancy_density_based(storage: StorageSuperset2, datapoints: List[str]):
    count_redundant = 0
    count_not_redundant = 0
    total = 0

    for idx, _ in enumerate(datapoints):
        datapoint = datapoints[idx]
        is_redundant = check_datapoint_density(storage, datapoint)

        total += 1
        if is_redundant:
            count_redundant += 1
            storage.remove_datapoint(datapoint)
            idx -= 1
        else:
            count_not_redundant += 1

    print("IN DJAKSTRA BASED FILTERING")
    print("Count not redundant", count_not_redundant)
    print("Count redundant", count_redundant)
    print("Total", total)


def filtering_redundancy_djakstra_based(storage: StorageSuperset2, datapoints: List[str]):
    count_redundant = 0
    count_not_redundant = 0
    total = 0

    for idx, _ in enumerate(datapoints):
        datapoint = datapoints[idx]
        is_connection_complete = check_datapoint_connections_completeness(storage, datapoint)
        is_redundant = False

        if not is_connection_complete:
            pass
        else:
            is_redundant = filtering_metric_djakstra(storage, datapoint)

        total += 1
        if is_redundant:
            count_redundant += 1
            storage.remove_datapoint(datapoint)
            idx -= 1
        else:
            count_not_redundant += 1

    print("IN DJAKSTRA BASED FILTERING")
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


def data_filtering_redundant_datapoints(storage: StorageSuperset2, verbose: bool = False):
    datapoints = storage.get_all_datapoints()
    # filtering_redundancy_djakstra_based(storage, datapoints)
    # filtering_redundancy_neighbors_based()
    filtering_redundancy_density_based(storage, datapoints)


def data_filtering_redundant_connections(storage: StorageSuperset2, verbose: bool = False):
    datapoints = storage.get_all_datapoints()
    count_redundant = 0
    total_count = 0

    total_count = len(storage.get_all_connections_data())
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
                if abs(first_angle - second_angle) < REDUNDANCY_CONNECTION_ANGLE:
                    # invalidate the bigger distance
                    if first_distance > second_distance:
                        to_remove.append(first_connection)
                    else:
                        to_remove.append(second_connection)

        for connection in to_remove:
            count_redundant += storage.remove_connection(datapoint, connection["end"])

    if verbose:
        print("IN CONNECTION REDUNDANCY FILTERING")
        print("Count redundant", count_redundant)
        print("Total connections before", total_count)
        print("Total connections after", len(storage.get_all_connections_data()))
