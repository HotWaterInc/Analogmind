# singleton for handling data
import json
from src.utils import get_project_root
from .parameters import CollectedDataType, Paths, get_data_file_path

def get_file_path(data_sample: CollectedDataType):
    if data_sample == CollectedDataType.Data8x8:
        return Paths.Data8x8.value
    elif data_sample == CollectedDataType.Data15x15:
        return Paths.Data15x15.value
    else:
        return None

class ExternalDataHandler:
    __instance = None

    data_array = []
    data_sample = CollectedDataType.Data8x8

    @staticmethod
    def get_instance():
        if ExternalDataHandler.__instance is None:
            ExternalDataHandler()
        return ExternalDataHandler.__instance

    def __init__(self):
        if ExternalDataHandler.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ExternalDataHandler.__instance = self
            self.data_array = []

    def set_data_sample(self, data_sample: CollectedDataType):
        self.data_sample = data_sample

    def append_data(self, data):
        self.data_array.append(data)

    def write_data_array_to_file(self)->None:
        data = self.data_array

        root_path = get_project_root()
        file_path = root_path + "/" + get_data_file_path(self.data_sample)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)  # indent=4 for pretty printing

    # Read data from file
    def read_data_array_from_file(self):

        root_path = get_project_root()
        file_path = root_path + "/" + get_data_file_path(self.data_sample)

        with open(file_path, 'r') as file:
            data_arr = json.load(file)
        return data_arr