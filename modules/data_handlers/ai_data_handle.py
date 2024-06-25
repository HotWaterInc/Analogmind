import json
from utils import get_project_root
from .parameters import DataSampleType, Paths, get_file_path

def read_data_array_from_file(self, data_sample: DataSampleType):
    root_path = get_project_root()
    file_path = root_path + "/" + get_file_path(self.data_sample)

    with open(file_path, 'r') as file:
        data_arr = json.load(file)
    return data_arr

