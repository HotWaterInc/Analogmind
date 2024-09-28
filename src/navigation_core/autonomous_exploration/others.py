from typing import List
from torch import nn
from src.runtime_storages.storage_struct import StorageStruct
from src.runtime_storages.types import ConnectionSyntheticData
from src import runtime_storages as storage


def fill_augmented_connections_directions(new_connections: List[ConnectionSyntheticData], storage_struct: StorageStruct,
                                          image_direction_network: nn.Module):
    pass
    # additional_connections_augmented: L = []
    #
    # start_data_array = []
    # end_data_array = []
    #
    # start_name_array = []
    # end_name_array = []
    # distances_array = []
    # index_array = []
    #
    # directions_thetas_hashmap = {}
    # SELECTIONS = 4
    # for idx, connection in enumerate(new_connections):
    #     start = connection["start"]
    #     end = connection["end"]
    #
    #     for i in range(SELECTIONS):
    #         i = random.randint(0, ROTATIONS - 1)
    #         start_data = storage_struct.node_get_datapoints_tensor(start)[i]
    #         end_data = storage_struct.node_get_datapoints_tensor(end)[i]
    #
    #         start_data_array.append(start_data)
    #         end_data_array.append(end_data)
    #         index_array.append(idx)
    #
    # start_data_array = torch.stack(start_data_array).to(get_device())
    # end_data_array = torch.stack(end_data_array).to(get_device())
    # direction_thetas = image_direction_network(start_data_array, end_data_array)
    #
    # for idx, direction_thetas in enumerate(direction_thetas):
    #     index_connection = index_array[idx]
    #     if index_connection not in directions_thetas_hashmap:
    #         directions_thetas_hashmap[index_connection] = torch.tensor(0.0, device=get_device())
    #     directions_thetas_hashmap[index_connection] += direction_thetas
    #
    # for idx in directions_thetas_hashmap:
    #     start_name = new_connections[idx]["start"]
    #     end_name = new_connections[idx]["end"]
    #     direction_thetas = directions_thetas_hashmap[idx] / SELECTIONS
    #     direction = direction_thetas_to_radians(direction_thetas)
    #
    #     distance_authenticity = new_connections[idx]["markings"]["distance"] == "authentic"
    #     distance = new_connections[idx]["distance"]
    #     direction = generate_dxdy(direction, distance)
    #
    #     new_connections = generate_connection(
    #         start=start_name,
    #         end=end_name,
    #         distance=distance,
    #         direction=direction,
    #         distance_authenticity=distance_authenticity,
    #         direction_authenticity=direction
    #     )
    #     additional_connections_augmented.append(new_connections)
    #
    # return additional_connections_augmented


def synthetic_connections_fill_directions(new_connections: List[ConnectionSyntheticData],
                                          storage_struct: StorageStruct):
    for idx, connection in enumerate(new_connections):
        start = connection["start"]
        end = connection["end"]
        direction = storage.get_direction_between_nodes_metadata(storage_struct, start, end)
        new_connections[idx]["direction"] = direction

    return new_connections


def synthetic_connections_fill_distances(new_connections: List[ConnectionSyntheticData],
                                         storage_struct: StorageStruct):
    for idx, connection in enumerate(new_connections):
        start = connection["start"]
        end = connection["end"]
        distance = storage.get_distance_between_nodes_metadata(storage_struct, start, end)
        new_connections[idx]["distance"] = distance

    return new_connections


def fill_augmented_connections_distances(additional_connections: List[any], storage_struct: StorageStruct,
                                         image_distance_network: nn.Module):
    pass
    # additional_connections_augmented = []
    #
    # start_data_array = []
    # end_data_array = []
    #
    # start_name_array = []
    # end_name_array = []
    # directions_array = []
    # index_array = []
    #
    # distances_hashmap = {}
    # SELECTIONS = 4
    # for idx, connection in enumerate(additional_connections):
    #     start = connection["start"]
    #     end = connection["end"]
    #
    #     for i in range(SELECTIONS):
    #         i = random.randint(0, ROTATIONS - 1)
    #         start_data = storage.node_get_datapoints_tensor(start)[i]
    #         end_data = storage.node_get_datapoints_tensor(end)[i]
    #
    #         start_data_array.append(start_data)
    #         end_data_array.append(end_data)
    #         index_array.append(idx)
    #
    # if len(start_data_array) == 0:
    #     return additional_connections_augmented
    #
    # start_data_array = torch.stack(start_data_array).to(get_device())
    # end_data_array = torch.stack(end_data_array).to(get_device())
    # distances = image_distance_network(start_data_array, end_data_array)
    #
    # for idx, dist in enumerate(distances):
    #     index_connection = index_array[idx]
    #     if index_connection not in distances_hashmap:
    #         distances_hashmap[index_connection] = 0
    #     distances_hashmap[index_connection] += dist.item()
    #
    # for idx in distances_hashmap:
    #     start_name = additional_connections[idx]["start"]
    #     end_name = additional_connections[idx]["end"]
    #     distance = distances_hashmap[idx] / SELECTIONS
    #     direction = additional_connections[idx]["direction"]
    #     direction_authenticity = additional_connections[idx]["markings"]["direction"] == "authentic"
    #
    #     new_connections = generate_connection(
    #         start=start_name,
    #         end=end_name,
    #         distance=distance,
    #         direction=direction,
    #         distance_authenticity=False,  # because it is generated
    #         direction_authenticity=direction_authenticity
    #     )
    #     additional_connections_augmented.append(new_connections)
    #
    # return additional_connections_augmented
