from sklearn.preprocessing import MinMaxScaler
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

