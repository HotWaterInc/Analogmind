import math
from src.modules.save_load_handlers.ai_data_handle import read_data_from_file, write_other_data_to_file
from src.modules.save_load_handlers.parameters import *
from src.ai.data_processing.ai_data_processing import process_adjacency_properties


def generate_grid_connections() -> None:
    """
    One timer used to generate connections between sensor positions based on metadata
    These connections would normally be collected in the environemnt during training

    Saves the connections to a file
    :return: None
    """
    json_data = read_data_from_file(CollectedDataType.Data8x8)
    all_sensor_data = [[item[DATA_SENSORS_FIELD], item[DATA_PARAMS_FIELD]["i"], item[DATA_PARAMS_FIELD]["j"]] for item
                       in json_data]
    process_adjacency_properties(all_sensor_data)
    indices_properties = []
    needed_data = [
        [item[DATA_SENSORS_FIELD], item[DATA_PARAMS_FIELD]["i"], item[DATA_PARAMS_FIELD]["j"], item[DATA_NAME_FIELD]]
        for item in json_data]

    connections_array = []

    # iterate indices
    for i in range(len(indices_properties)):
        if indices_properties[i][2] == 1:
            start_indice = indices_properties[i][0]
            end_indice = indices_properties[i][1]

            start_data = needed_data[start_indice]
            start_i = start_data[1]
            start_j = start_data[2]
            start_name = start_data[3]

            end_data = needed_data[end_indice]
            end_i = end_data[1]
            end_j = end_data[2]
            end_name = end_data[3]

            distance = math.sqrt((start_i - end_i) ** 2 + (start_j - end_j) ** 2)
            direction_vector = (end_i - start_i, end_j - start_j)
            normalized_direction_vector = (direction_vector[0] / distance, direction_vector[1] / distance)

            print(
                f"Start: {start_i}, {start_j} End: {end_i}, {end_j} Distance: {distance} Direction: {normalized_direction_vector}")
            connections_array.append({
                "start": start_name,
                "end": end_name,
                "distance": distance,
                "direction": normalized_direction_vector
            })

    write_other_data_to_file("data8x8_connections.json", connections_array)
