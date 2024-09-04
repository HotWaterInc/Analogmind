import time
import math
from typing import Dict, TypedDict, Generator, List
from src.action_ai_controller import ActionAIController
from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.save_load_handlers.data_handle import write_other_data_to_file

from src.action_robot_controller import detach_robot_sample_distance, detach_robot_sample_image, \
    detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute, \
    detach_robot_rotate_continuous_absolute, detach_robot_forward_continuous, detach_robot_sample_image_inference
import threading
import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.modules.save_load_handlers.ai_models_handle import save_ai, save_ai_manually, load_latest_ai, \
    load_manually_saved_ai
from src.modules.save_load_handlers.parameters import *
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.runtime_data_storage import Storage
from typing import List, Dict, Union
from src.utils import array_to_tensor
import sys
import time
import asyncio
from src.modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import configs
import threading
from src.utils import perror
from src.modules.external_communication.communication_interface import send_data, CommunicationInterface
from src.utils import get_instance
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative
import torch
import math
from scipy.stats import norm
import torch
from torchvision import models, transforms
from PIL import Image
import time
from src.modules.save_load_handlers.data_handle import read_other_data_from_file, write_other_data_to_file

from src.modules.policies.data_collection import get_position, get_angle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def load_everything():
    global storage, direction_network
    storage = StorageSuperset2()
    grid_dataset = 5

    storage.load_raw_data_from_others(f"data{grid_dataset}x{grid_dataset}_rotated24_image_embeddings.json")
    storage.load_raw_data_connections_from_others(f"data{grid_dataset}x{grid_dataset}_connections.json")


def build_pil_image_from_recv(image_array):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_tensor_preprocess = preprocess(image_array)

    return input_tensor_preprocess


def pil_tensor_to_resnet18_embedding(pil_tensor) -> torch.Tensor:
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


def squeeze_out_resnet_output(embedding):
    return embedding.squeeze(3).squeeze(2).squeeze(0)


def build_pil_image_from_path(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_preprocess = transform(image)

    return image_preprocess


from PIL import Image
import numpy as np


def visualize_image_from_tensor(tensor_data, save_name: str):
    try:
        # Ensure the input is a numpy array
        if not isinstance(tensor_data, np.ndarray):
            tensor_data = np.array(tensor_data)

        # Check if the shape is correct

        # Transpose the array to (height, width, channels)
        img_data = np.transpose(tensor_data, (1, 2, 0))

        # Normalize the data if it's not already in 0-255 range
        if img_data.max() <= 1.0:
            img_data = (img_data * 255).astype(np.uint8)

        # Create image from array
        img = Image.fromarray(img_data, mode='RGB')

        # Display basic information about the image
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")

        # Save the image to a file
        output_path = f"./{save_name}.jpg"
        img.save(output_path, 'JPEG')
        print(f"Image saved successfully to {output_path}")

        return img

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def webots_radians_to_normal(x: float) -> float:
    if x < 0:
        x += 2 * math.pi
    return x


def process_webots_image_to_embedding(webots_raw_image) -> torch.Tensor:
    webots_image_np = np.array(webots_raw_image)
    webots_image_np = webots_image_np.astype(np.uint8)

    if webots_image_np.shape[-1] == 3:
        webots_image_np = webots_image_np[:, :, [2, 1, 0]]

    pil_image = Image.fromarray(webots_image_np)

    input_batch_received = build_pil_image_from_recv(pil_image)
    emb1 = pil_tensor_to_resnet18_embedding(input_batch_received)

    return emb1


def test_images_accuracy():
    load_everything()
    width = 3
    height = 3
    grid_size = 5
    total_rotations = 24
    i, j = 0, 0
    rotation = 0
    x, y = get_position(width, height, grid_size, i, j, 0, 0.5)
    angle = get_angle(total_rotations, rotation)
    time.sleep(0.25)
    detach_robot_teleport_absolute(x, y)
    yield
    time.sleep(0.25)
    detach_robot_rotate_absolute(angle)
    yield
    detach_robot_sample_image_inference()
    yield

    global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
    buffer = global_data_buffer.buffer

    image_data = buffer["data"]
    empty_global_data_buffer()

    input_batch_path = build_pil_image_from_path("data/dataset5x5/image1.jpeg")

    emb1 = pil_tensor_to_resnet18_embedding(input_batch_path)
    emb2 = process_webots_image_to_embedding(image_data)
    emb1 = squeeze_out_resnet_output(emb1)
    emb2 = squeeze_out_resnet_output(emb2)

    print(torch.dist(emb1, emb2))
    embedding_json_00 = storage.get_datapoint_data_tensor_by_name("0_0")[0].to(device)
    print(torch.dist(emb1, embedding_json_00))
    print(torch.dist(emb2, embedding_json_00))
    yield


def build_resnet18_embeddings_CORRECT():
    json_data_ids = read_other_data_from_file("data5x5_rotated24_image_id.json")
    max_id = json_data_ids[-1]["data"][-1][0]
    print(f"Max id: {max_id}")

    json_index = 0
    j_index = 0

    batch_size = 100
    image_array = []

    image_path = "data/dataset5x5"
    for i in range(0, max_id + 1):
        full_path = image_path + f"/image{i + 1}.jpeg"
        image = build_pil_image_from_path(full_path)
        embedding = pil_tensor_to_resnet18_embedding(image)
        embedding = squeeze_out_resnet_output(embedding)

        data_index = i % 24
        json_index = i // 24
        json_data_ids[json_index]["data"][data_index] = embedding.tolist()

        if i % batch_size == 0:
            print(f"Batch {i // batch_size} processed")

    write_other_data_to_file("data5x5_rotated24_image_embeddings.json", json_data_ids)
