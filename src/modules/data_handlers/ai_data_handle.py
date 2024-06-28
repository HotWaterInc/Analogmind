import json
from src.utils import get_project_root, prefix_path_with_root
from .parameters import CollectedDataType, get_file_path

def read_data_array_from_file(data_sample: CollectedDataType):
    root_path = get_project_root()
    local_path = get_file_path(data_sample)
    file_path = prefix_path_with_root(local_path)

    with open(file_path, 'r') as file:
        data_arr = json.load(file)
    return data_arr

