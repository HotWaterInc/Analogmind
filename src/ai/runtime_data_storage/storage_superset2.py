import numpy as np
import torch
import random

from .storage_superset import *
from .storage import *
from typing import List
from src.ai.data_processing.ai_data_processing import normalize_data_min_max_super

import math
from scipy.stats import norm

from ...utils import get_device


def normalize_direction(direction):
    direction = torch.tensor(direction, dtype=torch.float32, device=get_device())
    l2_direction = torch.norm(direction, p=2, dim=0, keepdim=True)
    direction = direction / l2_direction
    return direction


def calculate_coords_distance(coords1, coords2):
    return math.sqrt((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2)


def build_thetas_2(true_theta, thetas_length):
    thetas = torch.zeros(thetas_length)
    true_theta_index = true_theta * (thetas_length)

    integer_index_left = int(true_theta_index)
    integer_index_right = integer_index_left + 1

    weight_left = 1 - (true_theta_index - integer_index_left)
    weight_right = 1 - weight_left

    FILL_DISTANCE = 1
    for i in range(FILL_DISTANCE):
        left_index = integer_index_left - i
        right_index = integer_index_right + i

        pdf_value = weight_left
        thetas[left_index] = pdf_value

        pdf_value = weight_right
        thetas[right_index] = pdf_value

    return thetas


def direction_to_degrees_atan(direction):
    y = direction[1]
    x = direction[0]

    # Calculate the angle in radians using atan2
    angle_rad = math.atan2(x, y)

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)

    # Normalize the angle to be between 0 and 360 degrees
    normalized_angle = (angle_deg + 360) % 360
    # account for weird representation
    normalized_angle = (360 - angle_deg) % 360

    # print(direction, normalized_angle)

    return normalized_angle


def angle_to_thetas(true_theta, thetas_length):
    thetas = torch.zeros(thetas_length)
    true_theta_index = true_theta * (thetas_length)
    integer_index_left = int(true_theta_index)
    integer_index_right = integer_index_left + 1

    FILL_DISTANCE = 5
    SD = 1.5
    for i in range(FILL_DISTANCE):
        left_index = integer_index_left - i
        right_index = integer_index_right + i

        pdf_value = norm.pdf(left_index, loc=true_theta_index, scale=SD)
        if left_index < 0:
            left_index = len(thetas) + left_index

        thetas[left_index] = pdf_value

        pdf_value = norm.pdf(right_index, loc=true_theta_index, scale=SD)
        if right_index >= len(thetas):
            right_index = right_index - len(thetas)
        thetas[right_index] = pdf_value

    return thetas


def radians_to_degrees(radians):
    return radians * 180 / np.pi


def atan2_to_standard_radians(atan2):
    if atan2 < 0:
        atan2 = atan2 + 2 * np.pi

    atan2 = 2 * np.pi - atan2
    return atan2


def degrees_to_percent(angle_degrees):
    return angle_degrees / 360


def coordinate_pair_to_radians_cursed_tranform(x_component, y_component):
    """
    wizard magic

    In webots we measure angle counter clockwise from 0 to 2pi
    Only positive range can be transformed easily in percents
    counter clockwise because of some webots weird behavior when setting the rotation

    So [0,1] is 0 degrees
    [-1,0] is 90 degrees
    [0,-1] is 180 degrees
    [1,0] is 270 degrees

    You can infer the rest
    """
    # atan2 = math.atan2(y_component, x_component)
    atan2_inversed = math.atan2(x_component, y_component)
    radians_only_positive = atan2_to_standard_radians(atan2_inversed)
    return radians_only_positive


def radians_to_percent(radians):
    return radians / (2 * np.pi)


_cache_distances = {}


def distance_thetas_to_distance_percent(thetas):
    length = len(thetas)
    step = 1 / length
    distance_result = 0
    for i in range(length):
        distance = i * step
        distance_result += distance * thetas[i]

    # the thetas arr should already be normalized but just in case
    l1_norm = torch.norm(thetas, p=1)
    distance_result /= l1_norm

    return distance_result


def distance_percent_to_distance_thetas(true_theta_percent, thetas_length):
    thetas = torch.zeros(thetas_length)
    if true_theta_percent >= 1:
        true_theta_percent = 0.99

    true_theta_index = true_theta_percent * thetas_length
    if true_theta_index in _cache_distances:
        return _cache_distances[true_theta_index]

    integer_index_left = int(true_theta_index)
    integer_index_right = integer_index_left + 1

    FILL_DISTANCE = 3
    SD = 1
    for i in range(FILL_DISTANCE):
        left_index = integer_index_left - i
        right_index = integer_index_right + i

        if left_index > 0:
            pdf_value = norm.pdf(left_index, loc=true_theta_index, scale=SD)
            thetas[left_index] = pdf_value

        if right_index < len(thetas):
            pdf_value = norm.pdf(right_index, loc=true_theta_index, scale=SD)
            thetas[right_index] = pdf_value

    l1_norm = torch.norm(thetas, p=1)
    thetas /= l1_norm
    _cache_distances[true_theta_index] = thetas
    return thetas


_cache_thetas = {}


def angle_radians_to_percent(angle_radians):
    return angle_radians / (2 * np.pi)


def angle_percent_to_thetas_normalized(true_theta_percent, thetas_length):
    thetas = torch.zeros(thetas_length)
    if true_theta_percent == 1:
        true_theta_percent = 0

    if true_theta_percent in _cache_thetas:
        return _cache_thetas[true_theta_percent]

    true_theta_index = true_theta_percent * thetas_length
    integer_index_left = int(true_theta_index)
    integer_index_right = integer_index_left + 1

    FILL_DISTANCE = 5
    SD = 1.5
    for i in range(FILL_DISTANCE):
        left_index = integer_index_left - i
        right_index = integer_index_right + i

        pdf_value = norm.pdf(left_index, loc=true_theta_index, scale=SD)
        if left_index < 0:
            left_index = len(thetas) + left_index

        thetas[left_index] = pdf_value

        pdf_value = norm.pdf(right_index, loc=true_theta_index, scale=SD)
        if right_index >= len(thetas):
            right_index = right_index - len(thetas)
        thetas[right_index] = pdf_value

    l1_norm = torch.norm(thetas, p=1)
    thetas /= l1_norm

    _cache_thetas[true_theta_percent] = thetas
    return thetas


def deg_to_rad(degrees):
    return degrees * math.pi / 180


def thetas_to_radians(thetas):
    # theta is cos x + i sin x
    lng = len(thetas)
    step = 360 / lng
    real_arr = []
    imag_arr = []
    for i in range(lng):
        degree = i * step
        radians = deg_to_rad(degree)
        # print(degree, radians)

        real = math.cos(radians)
        imag = math.sin(radians)

        real_arr.append(real)
        imag_arr.append(imag)

    real_sum = 0
    imag_sum = 0
    for i in range(lng):
        real_sum += real_arr[i] * thetas[i]
        imag_sum += imag_arr[i] * thetas[i]
        # print(f"Real: {real_arr[i]}, Imag: {imag_arr[i]}, Theta: {thetas[i]}")

    real_sum /= lng
    imag_sum /= lng

    angle = math.atan2(imag_sum, real_sum)
    if angle < 0:
        angle += 2 * math.pi

    return angle


class StorageSuperset2(StorageSuperset):
    """
    Storage meant to handle multiple list inside the data field
    [ [1, 2, 3], [4, 5, 6], [7, 8, 9] ] instead of [1,2,3]
    """

    permutor = None

    def __init__(self):

        super().__init__()

    def set_permutor(self, permutor):
        self.permutor = permutor

    def build_permuted_data_raw_with_thetas(self) -> None:
        """
        Returns the data point by its name
        """
        for index, datapoint in enumerate(self.raw_env_data):
            name = datapoint["name"]
            data_tensor = torch.tensor(np.array(datapoint["data"]), dtype=torch.float32, device=get_device())
            thetas_batch = []
            rotations_count = len(data_tensor)
            for index_rot in range(rotations_count):
                theta = index_rot / rotations_count
                thetas = build_thetas(theta, 36)
                thetas_batch.append(thetas)

            thetas_batch = torch.stack(thetas_batch).to(get_device())

            permuted_data: torch.Tensor = self.permutor(data_tensor, thetas_batch)

            permuted_data_raw = permuted_data.tolist()
            self.raw_env_data[index]["data"] = permuted_data_raw

            # rebuilds map with new values
        self._convert_raw_data_to_map()

    def build_permuted_data_12images(self) -> None:
        """
        Returns the data point by its name
        """
        for index, datapoint in enumerate(self.raw_env_data):
            name = datapoint["name"]
            new_data = []
            for offset in range(24):
                new_datapoint = self.get_point_rotations_with_full_info_set_offset_concatenated(name, 12, offset)
                new_data.append(new_datapoint)

            new_data = torch.tensor(new_data, dtype=torch.float32, device=get_device())

            permuted_data: torch.Tensor = self.permutor(new_data)
            permuted_data_raw = permuted_data.tolist()

            self.raw_env_data[index]["data"] = permuted_data_raw

        # rebuilds map with new values
        self._convert_raw_data_to_map()

    def build_permuted_data_raw_abstraction_autoencoder_manifold(self) -> None:
        """
        Returns the data point by its name
        """

        for index, datapoint in enumerate(self.raw_env_data):
            data_tensor = torch.tensor(np.array(datapoint["data"]), dtype=torch.float32, device=get_device())
            manifold_position = self.permutor.encoder_inference(data_tensor)
            if isinstance(manifold_position, tuple):
                manifold_position = manifold_position[0]

            permuted_data_raw = manifold_position.tolist()
            self.raw_env_data[index]["data"] = permuted_data_raw

        # rebuilds map with new values
        self._convert_raw_data_to_map()

    def build_permuted_data_raw_abstraction_block_1img(self) -> None:
        """
        Returns the data point by its name
        """

        for index, datapoint in enumerate(self.raw_env_data):
            name = datapoint["name"]
            data_tensor = torch.tensor(np.array(datapoint["data"]), dtype=torch.float32, device=get_device())
            positional_encoding, rotational_encoding = self.permutor.encoder_training(data_tensor)

            permuted_data_raw = positional_encoding.tolist()
            self.raw_env_data[index]["data"] = permuted_data_raw

        # rebuilds map with new values
        self._convert_raw_data_to_map()

    def build_permuted_data_raw(self) -> None:
        """
        Returns the data point by its name
        """
        for index, datapoint in enumerate(self.raw_env_data):
            name = datapoint["name"]
            data_tensor = torch.tensor(np.array(datapoint["data"]), dtype=torch.float32)
            permuted_data: torch.Tensor = self.permutor(data_tensor)

            permuted_data_raw = permuted_data.tolist()
            self.raw_env_data[index]["data"] = permuted_data_raw

        # rebuilds map with new values
        self._convert_raw_data_to_map()

    raw_env_data_permuted_choice: List[Dict[str, any]] = []
    raw_env_data_permuted_choice_map: Dict[str, any] = {}

    _index_cache: Dict[str, int] = {}

    def get_datapoint_name_by_index(self, index: int) -> str:
        """
        Returns the data point by its name
        """
        return self.raw_env_data[index]["name"]

    def get_datapoint_index_by_name(self, name: str) -> int:
        """
        Returns the data point by its name
        """
        if name in self._index_cache:
            return self._index_cache[name]

        for index, datapoint in enumerate(self.raw_env_data):
            if datapoint["name"] == name:
                self._index_cache[name] = index
                return index

    def build_permuted_data_random_rotations_rotation_N(self, N: int) -> None:
        """
        Returns the data point by its name
        """
        self.raw_env_data_permuted_choice = []
        self._permutation_metadata = {}
        self._permutation_metadata_array = []

        random_pick = N
        for datapoint in self.raw_env_data:
            datapoint_copy = datapoint.copy()

            name = datapoint["name"]
            data_raw: List[List[float]] = datapoint["data"]
            length = len(data_raw)
            random_index = random_pick

            datapoint_copy["data"] = data_raw[random_index]
            self.raw_env_data_permuted_choice.append(datapoint_copy)
            self._permutation_metadata[name] = random_index
            self._permutation_metadata_array.append(random_index)

        for datapoint in self.raw_env_data_permuted_choice:
            name: str = datapoint["name"]
            self.raw_env_data_permuted_choice_map[name] = datapoint

    def build_permuted_data_random_rotations_rotation0(self) -> None:

        """
        Returns the data point by its name
        """
        self.raw_env_data_permuted_choice = []
        self._permutation_metadata = {}
        self._permutation_metadata_array = []

        for datapoint in self.raw_env_data:
            datapoint_copy = datapoint.copy()

            name = datapoint["name"]
            data_raw: List[List[float]] = datapoint["data"]
            length = len(data_raw)
            random_index = 0

            datapoint_copy["data"] = data_raw[random_index]
            self.raw_env_data_permuted_choice.append(datapoint_copy)
            self._permutation_metadata[name] = random_index
            self._permutation_metadata_array.append(random_index)

        for datapoint in self.raw_env_data_permuted_choice:
            name: str = datapoint["name"]
            self.raw_env_data_permuted_choice_map[name] = datapoint

    def build_permuted_data_random_rotations_custom(self, arr_custom) -> None:
        """
        Returns the data point by its name
        """
        self.raw_env_data_permuted_choice = []
        self._permutation_metadata = {}
        self._permutation_metadata_array = []

        for datapoint in self.raw_env_data:
            datapoint_copy = datapoint.copy()

            name = datapoint["name"]

            data_raw: List[List[float]] = datapoint["data"]
            length = len(arr_custom)
            random_index = random.randint(0, length - 1)
            random_index = arr_custom[random_index]

            datapoint_copy["data"] = data_raw[random_index]
            self.raw_env_data_permuted_choice.append(datapoint_copy)
            self._permutation_metadata[name] = random_index
            self._permutation_metadata_array.append(random_index)

        for datapoint in self.raw_env_data_permuted_choice:
            name: str = datapoint["name"]
            self.raw_env_data_permuted_choice_map[name] = datapoint

    def build_permuted_data_random_rotations(self) -> None:
        """
        Returns the data point by its name
        """
        self.raw_env_data_permuted_choice = []
        self._permutation_metadata = {}
        self._permutation_metadata_array = []

        for datapoint in self.raw_env_data:
            datapoint_copy = datapoint.copy()

            name = datapoint["name"]

            data_raw: List[List[float]] = datapoint["data"]
            length = len(data_raw)
            random_index = random.randint(0, length - 1)

            datapoint_copy["data"] = data_raw[random_index]
            self.raw_env_data_permuted_choice.append(datapoint_copy)
            self._permutation_metadata[name] = random_index
            self._permutation_metadata_array.append(random_index)

        for datapoint in self.raw_env_data_permuted_choice:
            name: str = datapoint["name"]
            self.raw_env_data_permuted_choice_map[name] = datapoint

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
