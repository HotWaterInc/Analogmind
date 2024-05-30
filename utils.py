from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import json

def normalize_data_min_max(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def get_json_data(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    json_data = [parse_json_string(item) for item in json_data]
    return json_data

def parse_json_string(json_string):
    try:
        parsed_dict = json.loads(json_string)
        return parsed_dict
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None



def process_data():
    json_data = get_json_data('data.json')
    all_sensor_data = [[item['sensor_data'], item["i_index"], item["j_index"]] for item in json_data]
    sensor_data = [item['sensor_data'] for item in json_data]
    sensor_data = normalize_data_min_max(np.array(sensor_data))
    sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
    return all_sensor_data, sensor_data


def load_ai(name, network_type):
    autoencoder = network_type()
    autoencoder.load_state_dict(torch.load(name))
    return autoencoder
