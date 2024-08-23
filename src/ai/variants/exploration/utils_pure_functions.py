from typing import Dict
import numpy as np
import math


def get_distance_coords_pair(coords1: any, coords2: any) -> float:
    x1, y1 = coords1
    x2, y2 = coords2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_real_distance_between_datapoints(datapoint1: Dict[str, any], datapoint2: Dict[str, any]) -> float:
    coords1 = datapoint1["params"]["x"], datapoint1["params"]["y"]
    coords2 = datapoint2["params"]["x"], datapoint2["params"]["y"]
    return get_distance_coords_pair(coords1, coords2)


def get_direction_between_datapoints(datapoint1: Dict[str, any], datapoint2: Dict[str, any]) -> tuple[float, float]:
    coords1 = datapoint1["params"]["x"], datapoint1["params"]["y"]
    coords2 = datapoint2["params"]["x"], datapoint2["params"]["y"]
    direction_vector = (coords2[0] - coords1[0], coords2[1] - coords1[1])
    return direction_vector


def sample_n_elements(data: list, n: int) -> list:
    if n >= len(data):
        return data

    return np.random.choice(data, n, replace=False)
