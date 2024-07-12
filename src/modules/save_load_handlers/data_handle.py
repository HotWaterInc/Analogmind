import json
from src.utils import get_project_root, prefix_path_with_root
from .parameters import CollectedDataType, get_data_file_path


def read_data_from_file(data_sample: CollectedDataType):
    local_path = get_data_file_path(data_sample)
    file_path = prefix_path_with_root(local_path)

    with open(file_path, 'r') as file:
        data_arr = json.load(file)
    return data_arr


def read_other_data_from_file(file_name: str):
    data_sample = CollectedDataType.Other
    local_path = get_data_file_path(data_sample)
    file_path = prefix_path_with_root(local_path + file_name)

    with open(file_path, 'r') as file:
        data_arr = json.load(file)
    return data_arr


def write_data_to_file(data_sample: CollectedDataType, data_arr):
    local_path = get_data_file_path(data_sample)
    file_path = prefix_path_with_root(local_path)

    with open(file_path, 'w') as file:
        json.dump(data_arr, file, indent=4)


def write_other_data_to_file(file_name: str, data: any) -> None:
    data_sample = CollectedDataType.Other
    local_path = get_data_file_path(data_sample)
    file_path = prefix_path_with_root(local_path + file_name)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
