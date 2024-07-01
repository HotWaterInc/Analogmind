"""
Used to store the data that the AI works with.

Meant to apply optimizations to data processing and retrieval, as well as be an intermediary between the AI
and the data that flows from the environment

"""
import enum
from typing import List, Dict, Union
from typing_extensions import TypedDict
from src.modules.data_handlers.ai_data_handle import read_data_from_file, CollectedDataType, read_other_data_from_file
import numpy as np
import torch
from src.ai.data_processing.ai_data_processing import normalize_data_min_max


class RawEnvironmentData(TypedDict):
    """
    Represents the raw data that comes from the environment.

    'name' is a unique identifier for the data point
    """
    data: List[float]
    name: str  # uid of the datapoint
    params: Dict[str, any]


class DirectionCoord(enum.Enum):
    X = 0
    Y = 1


class RawConnectionData(TypedDict):
    start: str
    end: str
    distance: float
    direction: List[float]


class AdjacencyDataSample(TypedDict):
    start: str
    end: str
    distance: int


class Storage():
    """
    Meant as an intermediary between the AI and data processing / handling at runtime
    Any data including the one that might come continously from the environment will flow through here
    """
    raw_env_data: List[RawEnvironmentData] = []
    raw_connections_data: List[RawConnectionData] = []

    raw_env_data_map: Dict[str, RawEnvironmentData] = {}  # uid -> data

    custom_data: Dict[str, any] = {}

    def __init__(self):
        pass

    def _convert_raw_data_to_map(self):
        """
        Converts the raw data to a map for faster access
        """
        for item in self.raw_env_data:
            self.raw_env_data_map[item["name"]] = item

    def load_raw_data(self, data_type: CollectedDataType):
        """
        Loads the raw data from the file
        """
        if data_type == CollectedDataType.Data8x8:
            self.raw_env_data = read_data_from_file(data_type)
            self.raw_connections_data = read_other_data_from_file("data8x8_connections.json")
        else:
            raise ValueError("Data type not supported")

        self._convert_raw_data_to_map()

    def get_connections_data(self) -> List[RawConnectionData]:
        return self.raw_connections_data

    def _generate_adjacency_data_sample(self, item: RawConnectionData, distance: int = 1) -> AdjacencyDataSample:
        return {
            "start": item["start"],
            "end": item["end"],
            "distance": distance
        }

    def get_adjacency_data(self) -> List[AdjacencyDataSample]:
        return [self._generate_adjacency_data_sample(item, 1) for item in self.raw_connections_data]

    def add_custom_data(self, key: str, data: any):
        """
        Adds custom data to the storage
        """
        self.custom_data[key] = data

    _connections_numpy_array: np.ndarray = None

    def sample_adjacent_datapoints(self, sample_size: int) -> List[AdjacencyDataSample]:
        """
        Samples a number the adjacent datapoints

        :param sample_size: the number of datapoints to sample
        """
        # samples from connections since they are adjacent
        if self._connections_numpy_array is None:
            self._connections_numpy_array = np.array(self.raw_connections_data)

        sampled_connections = np.random.choice(self._connections_numpy_array, sample_size, replace=False)
        sampled_adjacencies = [self._generate_adjacency_data_sample(item) for item in sampled_connections]

        return sampled_adjacencies

    _non_adjacent_numpy_array: np.ndarray = None

    def build_non_adjacent_numpy_array_from_metadata(self):
        """
        Builds the numpy array for non-adjacent data
        """
        array: List[AdjacencyDataSample] = []
        for i in range(len(self.raw_env_data)):
            for j in range(i + 1, len(self.raw_env_data)):
                i_x, i_y = self.raw_env_data[i]["params"]["i"], self.raw_env_data[i]["params"]["j"]
                j_x, j_y = self.raw_env_data[j]["params"]["i"], self.raw_env_data[j]["params"]["j"]
                distance = abs(i_x - j_x) + abs(i_y - j_y)
                adjacency_sample = AdjacencyDataSample(start=self.raw_env_data[i]["name"],
                                                       end=self.raw_env_data[j]["name"], distance=distance)
                array.append(adjacency_sample)

        self._non_adjacent_numpy_array = np.array(array, dtype=AdjacencyDataSample)

    def sample_non_adjacent_datapoints(self, sample_size: int) -> List[AdjacencyDataSample]:
        """
        Samples a number of non-adjacent datapoints

        :param sample_size: the number of datapoints to sample
        """
        if self._non_adjacent_numpy_array is None:
            self.build_non_adjacent_numpy_array_from_metadata()

        sampled_connections = np.random.choice(self._non_adjacent_numpy_array, sample_size, replace=False)
        return sampled_connections

    def get_non_adjacent_data(self) -> List[AdjacencyDataSample]:
        """
        Returns all the non-adjacent data
        """
        if self._non_adjacent_numpy_array is None:
            self.build_non_adjacent_numpy_array_from_metadata()

        # eliminate item["distance"] that are 1
        return [self._generate_adjacency_data_sample(item, item["distance"]) for item in self._non_adjacent_numpy_array
                if item["distance"] > 1]

    def get_all_adjacent_data(self) -> List[AdjacencyDataSample]:
        """
        Returns all the non-adjacent data
        """
        if self._non_adjacent_numpy_array is None:
            self.build_non_adjacent_numpy_array_from_metadata()

        return [self._generate_adjacency_data_sample(item, item["distance"]) for item in self._non_adjacent_numpy_array]

    def get_non_adjacent_numpy_array(self):
        if self._non_adjacent_numpy_array is None:
            self.build_non_adjacent_numpy_array_from_metadata()

        return self._non_adjacent_numpy_array

    def get_pure_sensor_data(self):
        """
        Returns the sensor data from the environment, without any additional fields
        """
        return [item["data"] for item in self.raw_env_data]

    def get_datapoint_by_name(self, name: str):
        """
        Returns the data point by its name
        """
        return self.raw_env_data_map[name]

    _tensor_datapoints_data: Dict[str, torch.Tensor] = {}

    def get_data_tensor_by_name(self, name: str) -> torch.Tensor:
        """
        Returns the data point by its name
        """
        if name not in self._tensor_datapoints_data:
            self._tensor_datapoints_data[name] = torch.tensor(self.raw_env_data_map[name]["data"], dtype=torch.float32)

        return self._tensor_datapoints_data[name]

    def normalize_all_data(self):
        # normalizes all the data
        data = self.get_pure_sensor_data()
        normalized_data = normalize_data_min_max(np.array(data))

        length = len(self.raw_env_data)
        for i in range(length):
            self.raw_env_data[i]["data"] = normalized_data[i]
            name = self.raw_env_data[i]["name"]
            self.raw_env_data_map[name]["data"] = normalized_data[i]
