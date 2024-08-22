import numpy as np

from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage.storage import Storage, RawEnvironmentData
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from enum import Enum
from typing import List, Dict, Tuple
import torch
from src.utils import array_to_tensor


def pathfinding_step_super_ab(model: BaseAutoencoderModel, storage: StorageSuperset2, current_position_name: str,
                              target_position_name: str,
                              first_n_closest: int, max_search_distance: int) -> List[str]:
    """

    """
    # get closest pos within max_search_distance
    datapoints_within_distance: List[str] = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(
        current_position_name,
        max_search_distance)

    # !!! Assumes data is already preprocessed and normalized
    datapoints_sensor_data: List[RawEnvironmentData] = [storage.get_datapoint_by_name(datapoint) for datapoint in
                                                        datapoints_within_distance]

    datapoints_embeddings: List[Dict] = [{
        "embedding": model.encoder_inference(storage.get_datapoint_data_tensor_by_name_permuted(datapoint["name"])),
        "name": datapoint["name"]
    } for datapoint in datapoints_sensor_data]

    # take the n closest embeddings to the target embedding
    target_embedding = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(target_position_name))
    current_embedding = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(current_position_name))

    # tries to walk thought a continous latent space towards the target
    ab_step = target_embedding - current_embedding
    ab_step = ab_step / torch.norm(ab_step, p=2)

    next_embedding = current_embedding + ab_step * 2
    # finds closest datapoint to the next embedding
    distances = [torch.norm((datapoint["embedding"] - next_embedding), p=2).item() for datapoint in
                 datapoints_embeddings]
    sorted_datapoints = [datapoints_embeddings[i] for i in np.argsort(distances)]

    return [datapoint["name"] for datapoint in sorted_datapoints[:first_n_closest]]


def pathfinding_step_super(model: BaseAutoencoderModel, storage: StorageSuperset2, current_position_name: str,
                           target_position_name: str,
                           first_n_closest: int, max_search_distance: int) -> List[str]:
    """
    From the current position, finds the closest n positions to the target position (so the best n positions to go to)
    with a maximum distance of max_search_distance, and returns them

    :param model: The model used for embedding the sensor data
    :param current_position_name: The name of the current position
    :param target_position_name: The name of the target position
    :param storage: The storage object containing the data, used for retrieval
    :param first_n_closest: The number of closest positions to return
    :param max_search_distance: The maximum distance to search for positions
    :return: A list of the first n the closest positions

    """
    # get closest pos within max_search_distance
    datapoints_within_distance: List[str] = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(
        current_position_name,
        max_search_distance)

    # !!! Assumes data is already preprocessed and normalized
    datapoints_sensor_data: List[RawEnvironmentData] = [storage.get_datapoint_by_name(datapoint) for datapoint in
                                                        datapoints_within_distance]

    datapoints_embeddings: List[Dict] = [{
        "embedding": model.encoder_inference(storage.get_datapoint_data_tensor_by_name_permuted(datapoint["name"])),
        "name": datapoint["name"]
    } for datapoint in datapoints_sensor_data]

    # take the n closest embeddings to the target embedding
    target_embedding = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(target_position_name))

    distances = [torch.norm((datapoint["embedding"] - target_embedding), p=2).item() for datapoint in
                 datapoints_embeddings]
    sorted_datapoints = [datapoints_embeddings[i] for i in np.argsort(distances)]

    return [datapoint["name"] for datapoint in sorted_datapoints[:first_n_closest]]


def pathfinding_step(model: BaseAutoencoderModel, storage: Storage, current_position_name: str,
                     target_position_name: str,
                     first_n_closest: int, max_search_distance: int) -> List[str]:
    """
    From the current position, finds the closest n positions to the target position (so the best n positions to go to)
    with a maximum distance of max_search_distance, and returns them

    :param model: The model used for embedding the sensor data
    :param current_position_name: The name of the current position
    :param target_position_name: The name of the target position
    :param storage: The storage object containing the data, used for retrieval
    :param first_n_closest: The number of closest positions to return
    :param max_search_distance: The maximum distance to search for positions
    :return: A list of the first n the closest positions

    """
    # get closest pos within max_search_distance
    datapoints_within_distance: List[str] = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(
        current_position_name,
        max_search_distance)

    # !!! Assumes data is already preprocessed and normalized
    datapoints_sensor_data: List[RawEnvironmentData] = [storage.get_datapoint_by_name(datapoint) for datapoint in
                                                        datapoints_within_distance]

    datapoints_embeddings: List[Dict] = [{
        "embedding": model.encoder_inference(storage.get_datapoint_data_tensor_by_name(datapoint["name"])),
        "name": datapoint["name"]
    } for datapoint in datapoints_sensor_data]

    # take the n closest embeddings to the target embedding
    target_embedding = model.encoder_inference(storage.get_datapoint_data_tensor_by_name(target_position_name))

    distances = [torch.norm((datapoint["embedding"] - target_embedding), p=2).item() for datapoint in
                 datapoints_embeddings]
    sorted_datapoints = [datapoints_embeddings[i] for i in np.argsort(distances)]

    return [datapoint["name"] for datapoint in sorted_datapoints[:first_n_closest]]


def run_pathfinding(model: BaseAutoencoderModel, storage: Storage, start_position: str, target_position: str,
                    max_search_distance: int, first_n_closest: int):
    pass

#
# def run_lee(autoencoder, all_sensor_data, sensor_data):
#     starting_coords = (3, 3)
#     target_coords = (11, 10)
#
#     start_coords_data = []
#     end_coords_data = []
#
#     for i in range(len(all_sensor_data)):
#         i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
#         if i_x == starting_coords[0] and i_y == starting_coords[1]:
#             start_coords_data = sensor_data[i].unsqueeze(0)
#         if i_x == target_coords[0] and i_y == target_coords[1]:
#             end_coords_data = sensor_data[i].unsqueeze(0)
#
#     start_embedding = autoencoder.encoder_inference(start_coords_data)
#     end_embedding = autoencoder.encoder_inference(end_coords_data)
#
#     # take all adjacent coords
#     # calculate their embeddings
#     # take the closest adjacent embedding to the end embedding and "step" towards it ( as in go in that direction )
#     # repeat until the closest embedding is the end embedding
#
#     current_coords = starting_coords
#     explored_coords = []
#
#     while (current_coords != target_coords):
#         current_embedding = autoencoder.encoder_inference(sensor_data[i].unsqueeze(0))
#         closest_distance = 1000
#         closest_coords = None
#         for i in range(len(all_sensor_data)):
#             i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
#             # take only adjacent
#             abs_dist = abs(i_x - current_coords[0]) + abs(i_y - current_coords[1])
#             if abs_dist <= 2 and abs_dist > 0:
#                 i_embedding = autoencoder.encoder_inference(sensor_data[i].unsqueeze(0))
#                 distance = torch.norm((i_embedding - end_embedding), p=2).item()
#                 if distance < closest_distance:
#                     closest_distance = distance
#                     closest_coords = (i_x, i_y)
#
#         current_coords = closest_coords
#         explored_coords.append(current_coords)
#         print(f"Current coords: {current_coords}")
#         if current_coords == target_coords:
#             break
#
#
# def lee_improved_direction_step(autoencoder, current_position_name, target_position_name, json_data, connection_data):
#     # get embedding for current and target
#     current_position_data = find_position_data(json_data, current_position_name)
#     current_embedding = autoencoder.encoder_inference(current_position_data.unsqueeze(0))
#
#     target_position_data = find_position_data(json_data, target_position_name)
#     target_embedding = autoencoder.encoder_inference(target_position_data.unsqueeze(0))
#
#     connections = find_connections(current_position_name, connection_data)
#     conn_names = [connection[1] for connection in connections]
#     second_degree_connections = []
#     for conn_name in conn_names:
#         second_degree_connections += find_connections(conn_name, connection_data)
#
#     for connection in second_degree_connections:
#         start = connection[0]
#         end = connection[1]
#         # if start or end is not in conn_names, we add it
#         if start not in conn_names:
#             conn_names.append(start)
#         if end not in conn_names:
#             conn_names.append(end)
#
#     if current_position_name == "1_3":
#         print(connections)
#
#     if current_position_name == "1_3":
#         print(conn_names)
#
#     closest_points = [None, None, None]
#     closest_distances = [1000, 1000, 1000]
#
#     for conn_name in conn_names:
#         conn_data = find_position_data(json_data, conn_name)
#         conn_embedding = autoencoder.encoder_inference(conn_data.unsqueeze(0))
#         distance = torch.norm((conn_embedding - target_embedding), p=2).item()
#
#         if distance < closest_distances[0]:
#             closest_distances[2] = closest_distances[1]
#             closest_distances[1] = closest_distances[0]
#             closest_distances[0] = distance
#             closest_points[2] = closest_points[1]
#             closest_points[1] = closest_points[0]
#             closest_points[0] = conn_name
#         elif distance < closest_distances[1]:
#             closest_distances[2] = closest_distances[1]
#             closest_distances[1] = distance
#             closest_points[2] = closest_points[1]
#             closest_points[1] = conn_name
#         elif distance < closest_distances[2]:
#             closest_distances[2] = distance
#             closest_points[2] = conn_name
#
#     return closest_points
#
#
# def lee_direction_step_second_degree(autoencoder, storage, current_position_name, target_position_name, json_data,
#                                      connection_data):
#     # get embedding for current and target
#     current_position_data = find_position_data(json_data, current_position_name)
#     current_embedding = autoencoder.encoder_inference(current_position_data.unsqueeze(0))
#
#     target_position_data = find_position_data(json_data, target_position_name)
#     target_embedding = autoencoder.encoder_inference(target_position_data.unsqueeze(0))
#
#     connections = find_connections(current_position_name, connection_data)
#     conn_names = [connection[1] for connection in connections]
#     second_degree_connections = []
#     for conn_name in conn_names:
#         second_degree_connections += find_connections(conn_name, connection_data)
#
#     for connection in second_degree_connections:
#         start = connection[0]
#         end = connection[1]
#         # if start or end is not in conn_names, we add it
#         if start not in conn_names:
#             conn_names.append(start)
#         if end not in conn_names:
#             conn_names.append(end)
#
#     closest_point = None
#     closest_distances = 1000
#
#     for conn_name in conn_names:
#         conn_data = find_position_data(json_data, conn_name)
#         conn_embedding = autoencoder.encoder_inference(conn_data.unsqueeze(0))
#         distance = torch.norm((conn_embedding - target_embedding), p=2).item()
#
#         if distance < closest_distances:
#             closest_distances = distance
#             closest_point = conn_name
#
#     return closest_point
#
#
# def run_lee_improved(autoencoder, all_sensor_data, sensor_data):
#     # same as normal lee, but keeps a queue of current pairs, and for each one of them takes the 3 closes adjacent pairs and puts them in the queue
#     # if the current pair is the target pair, the algorithm stops
#
#     starting_coords = (2, 2)
#     target_coords = (2, 12)
#     start_coords_data = []
#     end_coords_data = []
#     banned_coords = [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]
#
#     for i in range(len(all_sensor_data)):
#         i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
#         if i_x == starting_coords[0] and i_y == starting_coords[1]:
#             start_coords_data = sensor_data[i].unsqueeze(0)
#         if i_x == target_coords[0] and i_y == target_coords[1]:
#             end_coords_data = sensor_data[i].unsqueeze(0)
#
#     start_embedding = autoencoder.encoder_inference(start_coords_data)
#     end_embedding = autoencoder.encoder_inference(end_coords_data)
#
#     # take all adjacent coords
#     # calculate their embeddings
#     # take the closest adjacent embedding to the end embedding and "step" towards it ( as in go in that direction )
#     # repeat until the closest embedding is the end embedding
#
#     current_coords = starting_coords
#     explored_coords = []
#
#     queue = [starting_coords]
#
#     while (current_coords != target_coords):
#         current_coords = queue.pop(0)
#         explored_coords.append(current_coords)
#         print(f"Current coords: {current_coords}")
#         if current_coords[0] == target_coords[0] and current_coords[1] == target_coords[1]:
#             return
#
#         closest_distances = [1000, 1000, 1000]
#         selected_coords = [[-1, -1], [-1, -1], [-1, -1]]
#
#         for i in range(len(all_sensor_data)):
#             i_x, i_y = all_sensor_data[i][1], all_sensor_data[i][2]
#             # take only adjacent
#             abs_dist = abs(i_x - current_coords[0]) + abs(i_y - current_coords[1])
#             if (i_x, i_y) in banned_coords:
#                 continue
#
#             if 0 < abs_dist <= 2:
#                 i_embedding = autoencoder.encoder_inference(sensor_data[i].unsqueeze(0))
#                 distance = torch.norm((i_embedding - end_embedding), p=2).item()
#
#                 if distance < closest_distances[0]:
#                     closest_distances[2] = closest_distances[1]
#                     closest_distances[1] = closest_distances[0]
#                     closest_distances[0] = distance
#                     selected_coords[2] = selected_coords[1]
#                     selected_coords[1] = selected_coords[0]
#                     selected_coords[0] = [i_x, i_y]
#                 elif distance < closest_distances[1]:
#                     closest_distances[2] = closest_distances[1]
#                     closest_distances[1] = distance
#                     selected_coords[2] = selected_coords[1]
#                     selected_coords[1] = [i_x, i_y]
#                 elif distance < closest_distances[2]:
#                     closest_distances[2] = distance
#                     selected_coords[2] = [i_x, i_y]
#
#                 # if distance < closest_distances[2]:
#                 #     closest_distances[2] = distance
#                 #     place_in_queue = True
#
#         # queue.append(selected_coords[0])
#         for selected_coord in selected_coords:
#             if selected_coord not in explored_coords and selected_coord not in queue:
#                 queue.append(selected_coord)
#
#         if current_coords == target_coords:
#             break
