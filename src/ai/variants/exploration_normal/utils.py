from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
import numpy as np


def check_min_distance(storage: StorageSuperset2, datapoint):
    datapoints_names = storage.get_all_datapoints()
    current_x = datapoint["params"]["x"]
    current_y = datapoint["params"]["y"]
    current_name = datapoint["name"]
    adjcent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 2)
    adjcent_names.append(current_name)

    minimum_real_distance = 1000000

    for name in datapoints_names:
        if name in adjcent_names or name == current_name:
            continue

        data = storage.get_datapoint_by_name(name)

        data_x = data["params"]["x"]
        data_y = data["params"]["y"]
        data_name = name

        real_distance = np.sqrt((current_x - data_x) ** 2 + (current_y - data_y) ** 2)
        if real_distance < minimum_real_distance:
            minimum_real_distance = real_distance

    return minimum_real_distance


def get_missing_connections_based_on_distance(storage: StorageSuperset2, datapoint, distance_threshold):
    datapoints_names = storage.get_all_datapoints()
    current_x = datapoint["params"]["x"]
    current_y = datapoint["params"]["y"]
    current_name = datapoint["name"]
    adjcent_names = storage.get_datapoint_adjacent_datapoints_at_most_n_deg(current_name, 2)
    adjcent_names.append(current_name)

    found_connections = []

    for name in datapoints_names:
        if name in adjcent_names or name == current_name:
            continue

        data = storage.get_datapoint_by_name(name)

        data_x = data["params"]["x"]
        data_y = data["params"]["y"]
        data_name = name

        real_distance = np.sqrt((current_x - data_x) ** 2 + (current_y - data_y) ** 2)
        if real_distance < distance_threshold:
            found_connections.append({
                "start": current_name,
                "end": data_name,
                "distance": real_distance,
                "direction": None
            })

    return found_connections
