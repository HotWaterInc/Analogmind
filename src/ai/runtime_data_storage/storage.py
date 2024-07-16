"""
Used to store the data that the AI works with.

Meant to apply optimizations to data processing and retrieval, as well as be an intermediary between the AI
and the data that flows from the environment

"""
import enum
from typing import List, Dict, Union, Tuple
from typing_extensions import TypedDict
from src.modules.save_load_handlers.data_handle import read_data_from_file, CollectedDataType, \
    read_other_data_from_file
import numpy as np
import torch
from src.ai.data_processing.ai_data_processing import normalize_data_min_max
from src.utils import perror, array_to_tensor


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


class Coords(TypedDict):
    x: int
    y: int


class RawConnectionData(TypedDict):
    start: str
    end: str
    distance: float
    direction: List[float]


class AdjacencyDataSample(TypedDict):
    start: str  # uid of the datapoint
    end: str  # uid of the datapoint
    distance: int


class Storage:
    """
    Meant as an intermediary between the AI and data processing / handling at runtime
    Any data including the one that might come continously from the environment will flow through here
    """
    metadata: Dict[str, any] = {
        "sensor_distance": 8,
    }
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

    def load_raw_data_from_others(self, filename: str):
        """
        Loads raw data from the others folder
        """
        self.raw_env_data = read_other_data_from_file(filename)
        self._convert_raw_data_to_map()

    def load_raw_data_connections_from_others(self, filename: str):
        """
        Loads raw data from the others folder
        """
        self.raw_connections_data = read_other_data_from_file(filename)

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

    def sample_adjacent_datapoints_connections_raw_data(self, sample_size: int) -> List[RawConnectionData]:
        """
        Samples a number the adjacent datapoints

        :param sample_size: the number of datapoints to sample
        """
        return np.random.choice(self.raw_connections_data, sample_size, replace=False)

    def sample_adjacent_datapoints_connections(self, sample_size: int) -> List[AdjacencyDataSample]:
        """
        Samples a number the adjacent datapoints

        :param sample_size: the number of datapoints to sample
        """
        # samples from connections since they are adjacent
        sampled_connections = np.random.choice(np.array(self.raw_connections_data), sample_size, replace=False)
        sampled_adjacencies = [self._generate_adjacency_data_sample(item) for item in sampled_connections]

        return sampled_adjacencies

    def sample_adjacent_datapoint_at_degree(self, datapoint_name: str, sample_size: int, degree: int) -> List[str]:
        """
        Samples a number of adjacent datapoints at a certain degree relative to datapoint name
        """

        adjacent_datapoints: List[str] = self.get_datapoints_adjacent_at_degree_n(datapoint_name, degree)

        sampled_adjacencies = np.random.choice(adjacent_datapoints, sample_size, replace=False)
        return sampled_adjacencies

    def sample_adjacent_datapoint_at_degree_most(self, datapoint_name: str, sample_size: int, degree: int) -> List[str]:
        """
        Samples a number of adjacent datapoints at a certain degree relative to datapoint name
        """

        adjacent_datapoints: List[str] = self.get_datapoint_adjacent_datapoints_at_most_n_deg(datapoint_name, degree)

        sampled_adjacencies = np.random.choice(adjacent_datapoints, sample_size, replace=False)
        return sampled_adjacencies

    def get_all_datapoints(self) -> List[str]:
        """
        Samples a number of random datapoints
        """
        return [item["name"] for item in self.raw_env_data]

    def sample_n_random_datapoints(self, sample_size: int) -> List[str]:
        """
        Samples a number of random datapoints
        """
        return np.random.choice([item["name"] for item in self.raw_env_data], sample_size, replace=False)

    _non_adjacent_numpy_array: np.ndarray = None

    def sample_triplet_anchor_positive_negative(self, anchor: str) -> Tuple[str, str]:
        """
        Samples an adjacent and non adjacent datapoints for a triplet loss
        """
        degree1_adjacent: List[RawConnectionData] = self.get_datapoint_adjacent_connections(anchor)
        # connections of degree 1
        degree1_adjacent: List[str] = [item["end"] for item in degree1_adjacent]

        degree1and2_adjacent: List[str] = self.get_datapoint_adjacent_datapoints_at_most_n_deg(anchor, 2)
        # connections of degree 2
        degree2_adjacent: List[str] = [item for item in degree1and2_adjacent if
                                       item not in degree1_adjacent]

        if len(degree1_adjacent) == 0:
            perror(f"Could not find adjacent connections for {anchor}")
            return None, None

        if len(degree2_adjacent) == 0:
            perror(f"Could not find non adjacent connections for {anchor}")
            return None, None

        deg1_adjacent: str = np.random.choice(degree1_adjacent)
        deg2_adjacent: str = np.random.choice(degree2_adjacent)

        return deg1_adjacent, deg2_adjacent

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

    def sample_datapoints_adjacencies(self, sample_size: int) -> List[AdjacencyDataSample]:
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

    def get_sensor_data_names(self):
        """
        Returns the sensor data names
        """
        return [item["name"] for item in self.raw_env_data]

    def get_raw_environment_data(self) -> List[RawEnvironmentData]:
        return self.raw_env_data

    def get_raw_connections_data(self) -> List[RawConnectionData]:
        return self.raw_connections_data

    def get_datapoint_by_name(self, name: str) -> RawEnvironmentData:
        """
        Returns the data point by its name
        """
        return self.raw_env_data_map[name]

    def get_datapoint_by_index(self, index: int) -> RawEnvironmentData:
        """
        Returns the data point by its index
        """
        return self.raw_env_data[index]

    def get_datapoint_data_tensor_by_name(self, name: str) -> torch.Tensor:
        """
        Returns the data point by its name
        """

        # return self._transformed_datapoints_data[name]
        return torch.tensor(self.raw_env_data_map[name]["data"], dtype=torch.float32)

    def get_datapoint_data_tensor_index_by_name(self, name: str) -> int:
        """
        Returns the index of the datapoint in the raw env data array, by its name
        """
        if name not in self.raw_env_data_map:
            return -1

        index = self.raw_env_data.index(self.raw_env_data_map[name])

        return index

    def tanh_all_data(self):
        # normalizes all the data
        data = self.get_pure_sensor_data()
        normalized_data = np.tanh(np.array(data))

        length = len(self.raw_env_data)
        for i in range(length):
            self.raw_env_data[i]["data"] = normalized_data[i]
            name = self.raw_env_data[i]["name"]
            self.raw_env_data_map[name]["data"] = normalized_data[i]

    def normalize_all_data(self):
        # normalizes all the data
        data = self.get_pure_sensor_data()
        normalized_data = normalize_data_min_max(np.array(data))

        length = len(self.raw_env_data)
        for i in range(length):
            self.raw_env_data[i]["data"] = normalized_data[i]
            name = self.raw_env_data[i]["name"]
            self.raw_env_data_map[name]["data"] = normalized_data[i]

    _connection_cache: Dict[str, List[RawConnectionData]] = {}

    def get_datapoint_adjacent_connections(self, datapoint_name: str) -> List[RawConnectionData]:
        """
        Returns the adjacent connections of a datapoint ( the connections that start or end with the datapoint )
        """
        found_connections = []
        connections_data = self.get_connections_data()
        # if datapoint_name in self._connection_cache:
        #     return self._connection_cache[datapoint_name]

        for connection in connections_data:
            connection_copy = connection.copy()
            start = connection_copy["start"]
            end = connection_copy["end"]
            distance = connection_copy["distance"]
            direction = connection_copy["direction"].copy()
            if start == datapoint_name:
                found_connections.append(connection_copy)
            if end == datapoint_name:
                # swap them
                direction[0] = -direction[0]
                direction[1] = -direction[1]

                aux = connection_copy["start"]
                connection_copy["start"] = connection_copy["end"]
                connection_copy["end"] = aux
                connection_copy["direction"] = direction

                found_connections.append(connection_copy)

        # self._connection_cache[datapoint_name] = found_connections
        return found_connections

    _connection_directed_cache: Dict[str, List[RawConnectionData]] = {}

    def get_datapoint_adjacent_connections_directed(self, datapoint_name: str) -> List[
        RawConnectionData]:
        """
        Gets only the connections which have the datapoint on the start field
        """
        found_connections = []
        if datapoint_name in self._connection_directed_cache:
            return self._connection_directed_cache[datapoint_name]

        connections_data = self.get_connections_data()
        for connection in connections_data:
            start = connection["start"]
            end = connection["end"]
            distance = connection["distance"]
            if start == datapoint_name:
                found_connections.append(connection)

        self._connection_directed_cache[datapoint_name] = found_connections
        return found_connections

    def _expand_existing_datapoints_with_adjacent(self, datapoints: List[str]):
        """
        Expands the datapoints with the adjacent ones
        """
        new_datapoints = []
        for datapoint in datapoints:
            connections = self.get_datapoint_adjacent_connections(datapoint)
            for connection in connections:
                start = connection["start"]
                end = connection["end"]
                if start == datapoint:
                    new_datapoints.append(end)
                if end == datapoint:
                    new_datapoints.append(start)

        # remove duplicates
        new_datapoints = list(set(new_datapoints))

        return new_datapoints

    def get_datapoint_adjacent_datapoints_at_most_n_deg(self, datapoint_name, distance_degree: int) -> List[str]:
        """
        Returns the connections of a datapoint that are at a certain distance degree from it
        """
        if distance_degree == 0:
            return [datapoint_name]

        found_data_points: List[str] = []
        found_data_points_map: Dict[str, bool] = {}
        new_data_points: List[str] = [datapoint_name]

        for degree in range(1, distance_degree + 1):
            # expands the datapoints with 1 layer of adjacent datapoints (1 degree unique datapoints)
            new_data_points = self._expand_existing_datapoints_with_adjacent(new_data_points)

            # checks for duplicates with the already found data points since it can also expand
            # inwards (we are interested only in outward)
            for new_data_point in new_data_points:
                if new_data_point not in found_data_points_map:
                    found_data_points_map[new_data_point] = True
                    found_data_points.append(new_data_point)

            found_data_points.extend(new_data_points)
        return found_data_points

    def get_datapoints_adjacent_at_degree_n(self, datapoint_name: str, degree: int) -> List[str]:
        """
        Returns the datapoints that are adjacent to a certain datapoint at a certain degree
        """
        adjacent_datapoints = self.get_datapoint_adjacent_datapoints_at_most_n_deg(datapoint_name, degree)
        adjacent_datapoints = [item for item in adjacent_datapoints if
                               item not in self.get_datapoint_adjacent_datapoints_at_most_n_deg(datapoint_name,
                                                                                                degree - 1)]

        return adjacent_datapoints

    _datapoints_coordinates_map: Dict[str, Coords] = {}

    def build_datapoints_coordinates_map(self):
        """
        Gets a map of datapoints names and their coordinates in a 2d space, based on connections data
        """
        datapoints_coordinates_map: Dict[str, Coords] = self._datapoints_coordinates_map
        explored_datapoints: Dict[str, bool] = {}

        # starts with first datapoint, could be any other one
        first_name = self.get_datapoint_by_index(0)["name"]
        x, y = 0, 0
        datapoints_coordinates_map[first_name] = Coords(x=x, y=y)

        # 0 is root, calculate further based on it
        # starting_datapoints = self.get_datapoint_adjacent_datapoints_at_most_n_deg(first_name, 1)
        queue: List[str] = [first_name]

        # gets datapoints internal pseudo xy mapping based on collected data
        while not len(queue) == 0:
            current_name = queue.pop(0)
            if current_name in explored_datapoints:
                continue

            explored_datapoints[current_name] = True
            # start is the current name
            connections = self.get_datapoint_adjacent_connections_directed(current_name)

            for connection in connections:
                end_name = connection["end"]

                # if position already found, we double-check if the calculated position matches the new calculated
                # position (they should be identical)
                if end_name in explored_datapoints:
                    x_start, y_start = x, y
                    distance = connection["distance"]
                    x_dir, y_dir = connection["direction"]
                    x_dir *= distance
                    y_dir *= distance
                    x_end, y_end = datapoints_coordinates_map[end_name]["x"], datapoints_coordinates_map[end_name]["y"]

                    if x_start + x_dir != x_end or y_start + y_dir != y_end:
                        perror(f"Found inconsistency at connection {current_name} to {end_name} inside storage")

                # if position not found, we calculate it and add it to the queue, as well as the map
                if end_name not in explored_datapoints:
                    x_dir, y_dir = connection["direction"]
                    distance = connection["distance"]

                    x_dir *= distance
                    y_dir *= distance
                    x_start = datapoints_coordinates_map[current_name]["x"]
                    y_start = datapoints_coordinates_map[current_name]["y"]
                    x_end = x_start + x_dir
                    y_end = y_start + y_dir

                    datapoints_coordinates_map[end_name] = Coords(x=x_end, y=y_end)
                    queue.append(end_name)

    def recenter_datapoints_coordinates_map(self):
        """
        Recenter the coordinates map so that the center of the coordinates is 0,0
        """
        datapoints_coordinates_map = self._datapoints_coordinates_map
        x_mean, y_mean = 0, 0
        total_datapoints = len(datapoints_coordinates_map)

        for key in datapoints_coordinates_map:
            x_mean += datapoints_coordinates_map[key]["x"]
            y_mean += datapoints_coordinates_map[key]["y"]

        x_mean /= total_datapoints
        y_mean /= total_datapoints

        for key in datapoints_coordinates_map:
            datapoints_coordinates_map[key]["x"] -= x_mean
            datapoints_coordinates_map[key]["y"] -= y_mean

    def get_datapoints_coordinates_map(self):
        return self._datapoints_coordinates_map
