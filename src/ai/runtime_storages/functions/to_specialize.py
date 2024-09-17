from typing import List, Dict


def get_adjacency_data(self) -> List[AdjacencyDataSample]:
    return [self._generate_adjacency_data_sample(item, 1) for item in self.raw_connections_data]


def sample_adjacent_datapoints_connections(self, sample_size: int) -> List[AdjacencyDataSample]:
    """
    Samples a number the adjacent datapoints

    :param sample_size: the number of datapoints to sample
    """
    # samples from connections since they are adjacent
    only_datapoints_connections = self.get_all_connections_only_datapoints()

    sampled_connections = np.random.choice(np.array(only_datapoints_connections), sample_size, replace=False)
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

    adjacent_datapoints: List[str] = self.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(datapoint_name,
                                                                                                    degree)

    sampled_adjacencies = np.random.choice(adjacent_datapoints, sample_size, replace=False)
    return sampled_adjacencies


_non_adjacent_numpy_array: np.ndarray = None


def sample_datapoints_adjacencies_cheated(self, sample_size: int) -> List[AdjacencyDataSample]:
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


def get_datapoint_adjacent_connections_cached(self, datapoint_name: str) -> List[RawConnectionData]:
    """
    Returns the adjacent connections of a datapoint ( the connections that start or end with the datapoint )
    """
    found_connections = []
    connections_data = self.get_all_connections_only_datapoints()
    if datapoint_name in self._connection_cache:
        return self._connection_cache[datapoint_name]

    for connection in connections_data:
        connection_copy = connection.copy()
        start = connection_copy["start"]
        end = connection_copy["end"]
        distance = connection_copy["distance"]
        if connection_copy["direction"] == None:
            continue
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

    self._connection_cache[datapoint_name] = found_connections
    return found_connections


def get_datapoint_adjacent_connections_null_connections(self, datapoint_name: str) -> List[RawConnectionData]:
    """
    Returns the adjacent connections of a datapoint ( the connections that start or end with the datapoint )
    """
    found_connections = []
    connections_data = self.get_all_connections_data()

    for connection in connections_data:
        connection_copy = connection.copy()
        start = connection_copy["start"]
        end = connection_copy["end"]
        if start == datapoint_name and end == None:
            found_connections.append(connection_copy)

    return found_connections


def get_datapoint_adjacent_connections_direction_filled(self, datapoint_name: str) -> List[RawConnectionData]:
    """
    Returns the adjacent connections of a datapoint ( the connections that start or end with the datapoint )
    """
    found_connections = []
    connections_data = self.get_all_connections_only_datapoints()

    for connection in connections_data:
        connection_copy = connection.copy()
        start = connection_copy["start"]
        end = connection_copy["end"]

        if start == datapoint_name:
            found_connections.append(connection_copy)

        if end == datapoint_name:
            direction = connection_copy["direction"].copy()
            # swap them
            if direction != None:
                direction[0] = -direction[0]
                direction[1] = -direction[1]

            aux = connection_copy["start"]
            connection_copy["start"] = connection_copy["end"]
            connection_copy["end"] = aux
            connection_copy["direction"] = direction

            found_connections.append(connection_copy)

    return found_connections


def get_datapoint_adjacent_connections(self, datapoint_name: str, null: bool = False) -> List[RawConnectionData]:
    """
    Returns the adjacent connections of a datapoint ( the connections that start or end with the datapoint )
    """
    found_connections = []
    connections_data = self.get_all_connections_only_datapoints()

    for connection in connections_data:
        connection_copy = connection.copy()
        start = connection_copy["start"]
        end = connection_copy["end"]

        if not null and end == None:
            continue

        if start == datapoint_name:
            found_connections.append(connection_copy)

        if end == datapoint_name:
            # swap them
            direction = connection_copy["direction"].copy()
            direction[0] = -direction[0]
            direction[1] = -direction[1]

            aux = connection_copy["start"]
            connection_copy["start"] = connection_copy["end"]
            connection_copy["end"] = aux
            connection_copy["direction"] = direction

            found_connections.append(connection_copy)

    return found_connections


def get_datapoint_adjacent_connections_non_null(self, datapoint_name: str) -> List[RawConnectionData]:
    """
    Returns the adjacent connections of a datapoint ( the connections that start or end with the datapoint )
    """
    found_connections = []
    connections_data = self.get_all_connections_only_datapoints()
    # if datapoint_name in self._connection_cache:
    #     return self._connection_cache[datapoint_name]

    for connection in connections_data:
        connection_copy = connection.copy()
        start = connection_copy["start"]
        end = connection_copy["end"]

        if start == None or end == None:
            continue

        if start == datapoint_name:
            found_connections.append(connection_copy)
        if end == datapoint_name:
            direction = connection_copy["direction"].copy()
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


def get_datapoint_adjacent_connections_directed(self, datapoint_name: str) -> List[
    RawConnectionData]:
    """
    Gets only the connections which have the datapoint on the start field
    """
    found_connections = []
    if datapoint_name in self._connection_directed_cache:
        return self._connection_directed_cache[datapoint_name]

    connections_data = self.get_all_connections_data()
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
        connections = self.get_datapoint_adjacent_connections(datapoint, null=False)
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


def _expand_existing_datapoints_with_adjacent_authentic(self, datapoints: List[str]):
    """
    Expands the datapoints with the adjacent ones
    """
    new_datapoints = []
    for datapoint in datapoints:
        connections = self.get_datapoint_adjacent_connections_authentic(datapoint)
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


def get_datapoints_adjacency_degree(self, datapoint1: str, datapoint2: str) -> int:
    """
    Returns the degree of adjacency between two datapoints
    """
    if datapoint1 == datapoint2:
        return 0

    degree = 1
    adjacent_datapoints = self.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(datapoint1, degree)
    while datapoint2 not in adjacent_datapoints:
        degree += 1
        adjacent_datapoints = self.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(datapoint1, degree)

    return degree


def get_datapoint_adjacent_datapoints_at_most_n_deg(self, datapoint_name, distance_degree: int) -> List[str]:
    """
    Returns the connections of a datapoint that are at a certain distance degree from it
    """
    if distance_degree == 0:
        return [datapoint_name]

    found_data_points: List[str] = []
    found_data_points_map: Dict[str, bool] = {}
    found_data_points_map[datapoint_name] = True

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

    return found_data_points


def get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(self, datapoint_name, distance_degree: int) -> List[
    str]:
    """
    Returns the connections of a datapoint that are at a certain distance degree from it
    """
    if distance_degree == 0:
        return [datapoint_name]

    found_data_points: List[str] = []
    found_data_points_map: Dict[str, bool] = {}
    found_data_points_map[datapoint_name] = True

    new_data_points: List[str] = [datapoint_name]

    for degree in range(1, distance_degree + 1):
        # expands the datapoints with 1 layer of adjacent datapoints (1 degree unique datapoints)
        new_data_points = self._expand_existing_datapoints_with_adjacent_authentic(new_data_points)

        # checks for duplicates with the already found data points since it can also expand
        # inwards (we are interested only in outward)
        for new_data_point in new_data_points:
            if new_data_point not in found_data_points_map:
                found_data_points_map[new_data_point] = True
                found_data_points.append(new_data_point)

    return found_data_points


def get_datapoints_adjacent_at_degree_n_as_raw_connection_data(self, datapoint_name: str, degree: int) -> List[
    RawConnectionData]:
    """
    Returns the datapoints that are adjacent to a certain datapoint at a certain degree
    """
    adjacent_degree_n = self.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(datapoint_name, degree)
    adjacent_degree_n_minus_1 = self.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(datapoint_name,
                                                                                               degree - 1)
    adjacent_data_points = [item for item in adjacent_degree_n if item not in adjacent_degree_n_minus_1]

    adjacent_at_deg_raw_connection_data = []
    for datapoint in adjacent_data_points:
        start = datapoint_name
        end = datapoint
        distance = degree
        direction = [0, 0]
        # calculate augmented direction
        start_data = self.get_datapoint_by_name(start)["params"]
        end_data = self.get_datapoint_by_name(end)["params"]

        x_start, y_start = start_data["x"], start_data["y"]
        x_end, y_end = end_data["x"], end_data["y"]

        x_dir = x_end - x_start
        y_dir = y_end - y_start
        direction = [x_dir, y_dir]

        connection_data = RawConnectionData(start=start, end=end, distance=distance, direction=direction)
        adjacent_at_deg_raw_connection_data.append(connection_data)

    return adjacent_at_deg_raw_connection_data


def get_datapoints_adjacent_at_degree_n(self, datapoint_name: str, degree: int) -> List[str]:
    """
    Returns the datapoints that are adjacent to a certain datapoint at a certain degree
    """
    adjacent_datapoints = self.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(datapoint_name, degree)
    adjacent_datapoints = [item for item in adjacent_datapoints if
                           item not in self.get_datapoint_adjacent_datapoints_at_most_n_deg_authentic(
                               datapoint_name,
                               degree - 1)]

    return adjacent_datapoints


_datapoints_coordinates_map: Dict[str, Coords] = {}


def build_sparse_datapoints_coordinates_map_based_on_xy(self, percent):
    raw_env = self.raw_env_data
    # sample only a percent of raw env data
    raw_env = np.random.choice(raw_env, int(len(raw_env) * percent), replace=False)

    for datapoint in raw_env:
        name = datapoint["name"]
        x = datapoint["params"]["x"]
        y = datapoint["params"]["y"]
        self._datapoints_coordinates_map[name] = Coords(x=x, y=y)


def build_datapoints_coordinates_map_based_on_xy(self):
    raw_env = self.raw_env_data
    for datapoint in raw_env:
        name = datapoint["name"]
        x = datapoint["params"]["x"]
        y = datapoint["params"]["y"]
        self._datapoints_coordinates_map[name] = Coords(x=x, y=y)


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
            if end_name == None:
                continue

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


def get_all_connections_possible_directions(self):
    datapoints = self.get_all_datapoints()

    for datapoint in datapoints:
        connections = self.get_datapoint_adjacent_connections_authentic(datapoint)
        for connection in connections:
            direction = connection["direction"]
            if direction == None:
                perror(f"Direction is None for connection {connection}")
                continue
            if direction[0] == 0 and direction[1] == 0:
                perror(f"Direction is 0 for connection {connection}")
                continue
