# singleton for handling data
import json
import enum
from utils import get_project_root

class DataSampleType(enum.Enum):
    Data8x8 = 1
    Data15x15 = 2

class Paths(enum.Enum):
    Data8x8 = 'data/data8x8.json'
    Data15x15 = 'data/data15x15.json'

def get_file_path(data_sample: DataSampleType):
    if data_sample == DataSampleType.Data8x8:
        return Paths.Data8x8.value
    elif data_sample == DataSampleType.Data15x15:
        return Paths.Data15x15.value
    else:
        return None

class ExternalDataHandler:
    __instance = None

    data_array = []
    data_sample = DataSampleType.Data8x8

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

    def set_data_sample(self, data_sample: DataSampleType):
        self.data_sample = data_sample

    def append_data(self, data):
        self.data_array.append(data)

    def write_data_array_to_file(self)->None:
        data = self.data_array

        root_path = get_project_root()
        file_path = root_path + "/" + get_file_path(self.data_sample)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)  # indent=4 for pretty printing

    # Read data from file
    def read_data_array_from_file(self):

        root_path = get_project_root()
        file_path = root_path + "/" + get_file_path(self.data_sample)

        with open(file_path, 'r') as file:
            data_arr = json.load(file)
        return data_arr