from src.navigation_core.pure_functions import direction_to_degrees_atan
from src.navigation_core.to_refactor.algorithms import build_connections_hashmap
from src.navigation_core.to_refactor.params import REDUNDANCY_CONNECTION_ANGLE
from src.runtime_storages.storage_struct import StorageStruct
import src.runtime_storages as storage
from typing import List


def filtering_metric_djakstra(storage_struct: StorageStruct, datapoint: str) -> any:
    connection_hashmap_without = build_connections_hashmap(storage_struct.get_all_connections_only_datapoints(),
                                                           [datapoint])

    connections_neighbors = storage_struct.connections_all_get(datapoint)
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


def filtering_redundancy_density_based(storage_struct: StorageStruct, datapoints: List[str]):
    count_redundant = 0
    count_not_redundant = 0
    total = 0

    for idx, _ in enumerate(datapoints):
        datapoint = datapoints[idx]
        is_redundant = check_datapoint_density(storage_struct, datapoint)

        total += 1
        if is_redundant:
            count_redundant += 1
            storage_struct.remove_datapoint(datapoint)
            idx -= 1
        else:
            count_not_redundant += 1

    print("IN DJAKSTRA BASED FILTERING")
    print("Count not redundant", count_not_redundant)
    print("Count redundant", count_redundant)
    print("Total", total)


def filtering_redundancy_djakstra_based(storage_struct: StorageStruct, datapoints: List[str]):
    count_redundant = 0
    count_not_redundant = 0
    total = 0

    for idx, _ in enumerate(datapoints):
        datapoint = datapoints[idx]
        is_connection_complete = check_datapoint_connections_completeness(storage_struct, datapoint)
        is_redundant = False

        if not is_connection_complete:
            pass
        else:
            is_redundant = filtering_metric_djakstra(storage_struct, datapoint)

        total += 1
        if is_redundant:
            count_redundant += 1
            storage_struct.remove_datapoint(datapoint)
            idx -= 1
        else:
            count_not_redundant += 1

    print("IN DJAKSTRA BASED FILTERING")
    print("Count not redundant", count_not_redundant)
    print("Count redundant", count_redundant)
    print("Total", total)


def filtering_redundancy_neighbors_based(storage_struct: StorageStruct, datapoints: List[str]):
    pass


def data_filtering_redundancies(storage_struct: StorageStruct):
    """
    Filter redundant datapoints which don't add any value to the topological graph
    or enhance the neural networks in any way
    """

    datapoints = storage_struct.get_all_datapoints()
    # filtering_redundancy_djakstra_based(storage_struct, datapoints)
    # filtering_redundancy_neighbors_based()


def data_filtering_redundant_datapoints(storage_struct: StorageStruct, verbose: bool = False):
    datapoints = storage_struct.get_all_datapoints()
    # filtering_redundancy_djakstra_based(storage_struct, datapoints)
    # filtering_redundancy_neighbors_based()
    filtering_redundancy_density_based(storage_struct, datapoints)


def filtering_redundant_connections(storage_struct: StorageStruct, verbose: bool = False):
    nodes: List[str] = storage.nodes_get_all_names(storage_struct)
    count_redundant = 0
    total_count = 0

    total_count = len(nodes)
    for node in nodes:
        connections = storage.node_get_connections_adjacent(storage_struct, node)
        connections_count = len(connections)
        to_remove = []
        for idx in range(connections_count):
            for jdx in range(idx + 1, connections_count):
                # possibly has faulty indices after removal, but it doesn't matter if the function is run many times
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
                    first_authentic = storage.connections_authentic_check_if_exists(storage_struct,
                                                                                    first_connection["start"],
                                                                                    first_connection["end"])
                    second_authentic = storage.connections_authentic_check_if_exists(storage_struct,
                                                                                     second_connection["start"],
                                                                                     second_connection["end"])
                    if not first_authentic and not second_authentic:
                        # we don't want to remove authentic connections, but we can get rid of synthetic ones
                        continue

                    if first_authentic and not second_authentic:
                        to_remove.append(first_connection)
                    elif not first_authentic and second_authentic:
                        to_remove.append(second_connection)
                    elif first_distance > second_distance:
                        to_remove.append(first_connection)
                    else:
                        to_remove.append(second_connection)

        for connection in to_remove:
            pass

            # count_redundant += storage_struct.remove_connection(node, connection["end"])

    if verbose:
        print("IN CONNECTION REDUNDANCY FILTERING")
        print("Count redundant", count_redundant)
        print("Total connections before", total_count)
        print("Total connections after", len(storage_struct.get_all_connections_data()))
