from typing import List, Dict
from src.ai.runtime_storages.storage_struct import StorageStruct
from src.ai.runtime_storages.visualization_data_struct import VisualizationDataStruct


def get_point_rotations_with_full_info_set_offset_concatenated(self, name: str, rotation_count: int, offset: int):
    """
    Returns the data point by its name
    """
    data_length = len(self.get_datapoint_data_by_name(name))
    new_data = self.get_point_rotations_with_full_info(name, rotation_count, offset)
    np_arr = np.array(new_data)
    return np_arr.flatten()


def get_point_rotations_with_full_info_random_offset_concatenated(self, name: str, rotation_count: int):
    """
    Returns the data point by its name
    """
    data_length = len(self.get_datapoint_data_by_name(name))
    offset = random.randint(0, data_length - 1)
    new_data = self.get_point_rotations_with_full_info(name, rotation_count, offset)
    np_arr = np.array(new_data)
    return np_arr.flatten()


def get_point_rotations_with_full_info_random_offset(self, name: str, rotation_count: int):
    """
    Returns the data point by its name
    """
    data_length = len(self.get_datapoint_data_by_name(name))
    offset = random.randint(0, data_length - 1)
    return self.get_point_rotations_with_full_info(name, rotation_count, offset)


def get_point_rotations_with_full_info(self, name: str, rotation_count: int, offset: int = 0) -> List[List[float]]:
    """
    Returns the data point by its name
    """
    data_raw: List[List[float]] = self.get_datapoint_data_by_name(name)
    rotation_total = len(data_raw)
    step = rotation_total / rotation_count
    new_data = []
    for i in range(rotation_count):
        index = int(i * step) + offset
        index = index % rotation_total
        new_data.append(data_raw[index])

    return new_data


def select_random_rotations_for_permuted_data(self):
    """
    For each datapoint in the transformed data, select a random sample to train the network on
    """
    arr = []
    for datapoint in self.raw_env_data:
        name = datapoint["name"]
        data_tensor = self._transformed_datapoints_data[name]
        length = len(data_tensor)
        random_index = random.randint(0, length - 1)
        selected_data = data_tensor[random_index]
        arr.append(selected_data.tolist())
    return arr


def get_pure_permuted_raw_env_data_with_xy_thresholds(self, x_threshold, y_threshold):
    """
    Returns the data point by its name
    """
    return [datapoint["data"] for datapoint in self.raw_env_data_permuted_choice if
            datapoint["params"]["x"] < x_threshold and datapoint["params"]["y"] < y_threshold]


def get_pure_permuted_raw_env_data(self):
    """
    Returns the data point by its name
    """
    return [datapoint["data"] for datapoint in self.raw_env_data_permuted_choice]


def get_pure_xy_permuted_raw_env_data(self):
    return [[datapoint["params"]["x"], datapoint["params"]["y"]] for datapoint in self.raw_env_data_permuted_choice]


_permutation_metadata: Dict[str, any] = {}
_permutation_metadata_array: List[int] = []


def get_pure_permuted_raw_env_metadata_array_rotation(self):
    return self._permutation_metadata_array


def get_number_of_permutations(self):
    return len(self.raw_env_data[0]["data"])


def get_datapoint_data_tensor_by_name_permuted(self, name: str) -> torch.Tensor:
    """
    Returns the data point by its name
    """
    return torch.tensor(self.raw_env_data_permuted_choice_map[name]["data"], dtype=torch.float32)


def incorporate_new_data(self, new_datapoints, new_connections):
    for datapoint in new_datapoints:
        self.raw_env_data.append(datapoint)

    for connection in new_connections:
        self.raw_connections_data.append(connection)

    self._convert_raw_data_to_map()


def connection_fill_synthetic_distances(self, distance_network, debug: bool = False):
    """
    Augments all synthetic or non-authentic connections with distances
    """
    distance_network = distance_network.to(get_device())
    distance_network.eval()

    connections = self.raw_connections_data
    SELECTIONS = 4
    err = 0

    start_data_arr = []
    end_data_arr = []
    real_distances = []  # for debugging
    indexes = []

    for idx, connection in enumerate(connections):
        if connection["distance"] is not None:
            continue

        start = connection["start"]
        end = connection["end"]
        direction = connection["direction"]
        distance = connection["distance"]
        real_distance = None

        for i in range(SELECTIONS):
            i = random.randint(0, ROTATIONS - 1)
            start_data = self.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(start, i)
            end_data = self.get_datapoint_data_selected_rotation_tensor_by_name_with_noise(end, i)
            index = idx

            start_data_arr.append(start_data)
            end_data_arr.append(end_data)
            indexes.append(index)

    start_data_tensor = torch.stack(start_data_arr).to(get_device())
    end_data_tensor = torch.stack(end_data_arr).to(get_device())
    synthetic_distances = distance_network(start_data_tensor, end_data_tensor)
    synthetic_distances_hashmap = {}

    for idx, synthetic_distance in enumerate(synthetic_distances):
        index = indexes[idx]
        if index not in synthetic_distances_hashmap:
            synthetic_distances_hashmap[index] = 0

        synthetic_distances_hashmap[index] += synthetic_distance

    pred_dist = []
    for hash_index in synthetic_distances_hashmap:
        synthetic_distances_hashmap[hash_index] /= SELECTIONS

        connection = connections[hash_index]
        connection["distance"] = synthetic_distances_hashmap[hash_index]
        pred_dist.append(synthetic_distances_hashmap[hash_index])

        if debug:
            real_distance = get_real_distance_between_datapoints(self.get_datapoint_by_name(connection["start"]),
                                                                 self.get_datapoint_by_name(connection["end"]))
            real_distances.append(real_distance)

    if debug:
        # debugging purposes again
        pred_dist = torch.tensor(pred_dist)
        real_distances = torch.tensor(real_distances)
        mse_loss = torch.nn.MSELoss()
        err = mse_loss(pred_dist, real_distances)
        print("Error for synthetically generated distances", err.item())


_non_adjacent_distances: Dict[str, Dict[str, float]] = None


def build_non_adjacent_distances_from_connections(self, debug: bool = False):
    """
    Builds the numpy array for non-adjacent data
    """
    if debug:
        print("STARTED BUILDING NON ADJCENT FLOYD CONNECTIONS")
        print("TOTAL CONNECTIONS WITH NULLITY", len(self.get_all_connections_data()))
        print("TOTAL CONNECTIONS", len(self.get_all_connections_only_datapoints()))
    connections_only_datapoints = self.get_all_connections_only_datapoints_authenticity_filter()

    if debug:
        print("FLOYD CONNECTIONS", len(connections_only_datapoints))
    connection_hashmap = build_connections_hashmap(connections_only_datapoints, [])
    distances = floyd_warshall_algorithm(connection_hashmap)
    if debug:
        print("")
        print("FINISHED WARSHALL")

    self._non_adjacent_distances = distances

    # iterates all datapoints
    datapoints = self.get_all_datapoints()
    length = len(datapoints)
    array = []
    total_error = 0
    total_error_samples = 0

    for start in range(length):
        for end in range(start + 1, length):
            start_name = datapoints[start]
            end_name = datapoints[end]
            distance = distances[start_name][end_name]
            adjacency_sample = AdjacencyDataSample(start=start_name, end=end_name, distance=distance)
            array.append(adjacency_sample)
            # additional checks
            if debug and random.randint(0, int(length * length / 100)) == 0:
                # print(f"Floyd warshall Distance between {start_name} and {end_name} is {distance}")
                distance_found = find_minimum_distance_between_datapoints_on_graph_djakstra(start_name, end_name,
                                                                                            connection_hashmap)
                # print("Alternative distance", distance_found)
                real_distance = self.get_datapoints_real_distance(start_name, end_name)
                # print("Real distance", real_distance)
                total_error += (distance - real_distance) ** 2
                total_error_samples += 1

    if debug and total_error_samples > 0:
        print("Average error", total_error / total_error_samples)
        print("Total error samples", total_error_samples)

    self._non_adjacent_numpy_array = np.array(array, dtype=AdjacencyDataSample)


def sample_datapoints_adjacencies(self, sample_size: int) -> List[AdjacencyDataSample]:
    """
    Samples a number of non-adjacent datapoints

    :param sample_size: the number of datapoints to sample
    """
    if self._non_adjacent_numpy_array is None:
        self.build_non_adjacent_distances_from_connections(debug=False)

    if sample_size > len(self._non_adjacent_numpy_array):
        warnings.warn(f"Sample size {sample_size} is larger than the non-adjacent datapoints array size")
        sample_size = len(self._non_adjacent_numpy_array)

    sampled_connections = np.random.choice(self._non_adjacent_numpy_array, sample_size, replace=False)
    return sampled_connections


def remove_datapoint(self, name: str):
    """
    Removes a datapoint by its name
    """
    associated_connections = []

    associated_connections = self.get_datapoint_adjacent_connections(name)
    for connection in associated_connections:
        self.remove_connection(connection["start"], connection["end"])

    self.remove_null_connections(name)

    for idx, datapoint in enumerate(self.raw_env_data):
        if datapoint["name"] == name:
            self.raw_env_data.pop(idx)
            break

    self._convert_raw_data_to_map()


def check_position_is_known_cheating(self, current_coords):
    """
    Returns if the position is known
    """
    datapoints = self.get_all_datapoints()
    for datapoint in datapoints:
        coords = self.get_datapoint_metadata_coords(datapoint)
        if calculate_coords_distance(coords, current_coords) < STEP_DISTANCE_CLOSE_THRESHOLD:
            return True

    return False


def get_datapoints_walking_distance(self, start_name, end_name):
    """
    Returns the walking distance between two datapoints
    """
    if self._non_adjacent_distances is None:
        self.build_non_adjacent_distances_from_connections(debug=False)

    return self._non_adjacent_distances[start_name][end_name]
