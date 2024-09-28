from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

import math
import torch

from src.agent_communication.response_data_buffer_class import AgentResponseDataBuffer, response_data_empty_buffer
from src.navigation_core.autonomous_exploration.params import STEP_DISTANCE_LOWER_BOUNDARY, STEP_DISTANCE_UPPER_BOUNDARY
from src.navigation_core.pure_functions import webots_radians_to_normal
from src.utils.utils import get_device


def random_direction_generator() -> float:
    return np.random.uniform(0, 2 * math.pi)


def random_distance_generator():
    return np.random.uniform(STEP_DISTANCE_LOWER_BOUNDARY, STEP_DISTANCE_UPPER_BOUNDARY)


def get_collected_data_distances() -> tuple[torch.Tensor, float, list[float]]:
    global_data_buffer: AgentResponseDataBuffer = AgentResponseDataBuffer.get_instance()
    buffer = global_data_buffer.buffer
    distances = buffer["data"]
    response_data_empty_buffer()

    angle = buffer["params"]["angle"]
    x = buffer["params"]["x"]
    y = buffer["params"]["y"]
    coords = [
        round(x, 3),
        round(y, 3)
    ]
    # WEBOTS
    angle = webots_radians_to_normal(angle)

    return distances, angle, coords


def get_collected_data_image() -> tuple[torch.Tensor, float, list[float]]:
    global_data_buffer: AgentResponseDataBuffer = AgentResponseDataBuffer.get_instance()
    buffer = global_data_buffer.buffer
    image_data = buffer["data"]
    response_data_empty_buffer()

    nd_array_data = np.array(image_data)
    angle = buffer["params"]["angle"]
    x = buffer["params"]["x"]
    y = buffer["params"]["y"]
    coords = [
        round(x, 3),
        round(y, 3)
    ]
    # trim coords to 3rd decimal

    angle = webots_radians_to_normal(angle)

    current_embedding = process_webots_image_to_embedding(nd_array_data).to(get_device())
    current_embedding = squeeze_out_resnet_output(current_embedding)

    return current_embedding, angle, coords


def process_webots_image_to_embedding(webots_raw_image) -> torch.Tensor:
    webots_image_np = np.array(webots_raw_image)
    webots_image_np = webots_image_np.astype(np.uint8)

    if webots_image_np.shape[-1] == 3:
        webots_image_np = webots_image_np[:, :, [2, 1, 0]]

    pil_image = Image.fromarray(webots_image_np)

    input_batch_received = _build_pil_image_from_recv(pil_image)
    emb1 = _pil_tensor_to_resnet18_embedding(input_batch_received)

    return emb1


def squeeze_out_resnet_output(embedding):
    return embedding.squeeze(3).squeeze(2).squeeze(0)


def _build_pil_image_from_recv(image_array):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_tensor_preprocess = preprocess(image_array)

    return input_tensor_preprocess


def _pil_tensor_to_resnet18_embedding(pil_tensor) -> torch.Tensor:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    preprocess2 = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor_preprocess = preprocess2(pil_tensor)
    # visualize_image_from_tensor(input_tensor, "image_recv_webots")
    input_tensor = preprocess2(input_tensor_preprocess)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    # Disable gradient calculation for inference
    embedding = None
    with torch.no_grad():
        # Get the embedding
        embedding = model(input_batch)

    return embedding


def _distance_sensors_transform(distance):
    # formula is roughly sensor_distance = 10 * distance + 2.5, for webots simulator
    return (distance - 2.5) / 10


def check_direction_validity(distance: float, direction: float, distance_sensors: any):
    # works only for north, needs adaptation for full rotation
    direction_percentage = direction / (2 * math.pi)
    sensors_count = len(distance_sensors)
    sensor_index_left = int(direction_percentage * sensors_count)
    sensor_index_right = (sensor_index_left + 1) % sensors_count
    wideness = 4

    for offset in range(wideness):
        left_index = sensor_index_left - offset
        right_index = sensor_index_right + offset

        if left_index < 0:
            left_index = sensors_count + left_index

        if right_index >= sensors_count:
            right_index = right_index - sensors_count

        sensor_left_distance = _distance_sensors_transform(distance_sensors[left_index])
        sensor_right_distance = _distance_sensors_transform(distance_sensors[right_index])

        if sensor_left_distance < distance or sensor_right_distance < distance:
            return False

    return True
