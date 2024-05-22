# singleton for handling data
import json

class DataHandling:
    __instance = None

    data_arr = []

    @staticmethod
    def get_instance():
        if DataHandling.__instance is None:
            DataHandling()
        return DataHandling.__instance

    def __init__(self):
        if DataHandling.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DataHandling.__instance = self
            self.data_arr = []

    def append_data(self, data):
        self.data_arr.append(data)

    def write_to_file(self, file_path)->None:
        data = self.data_arr
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)  # indent=4 for pretty printing

    # Read data from file
    def read_from_file(file_path):
        with open(file_path, 'r') as file:
            data_arr = json.load(file)
        return data_arr
