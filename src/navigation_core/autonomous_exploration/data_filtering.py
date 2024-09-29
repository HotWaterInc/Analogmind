from src.navigation_core.pure_functions import direction_to_degrees_atan
from src.navigation_core.to_refactor.algorithms import build_connections_hashmap
from src.navigation_core.to_refactor.params import REDUNDANCY_CONNECTION_ANGLE
from src.runtime_storages.storage_struct import StorageStruct
import src.runtime_storages as storage
from typing import List


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
