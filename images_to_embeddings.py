import sys
import time
import asyncio
from src.modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import configs
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
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


def build_resnet18_embeddings(image_path):
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    json_data_ids = read_other_data_from_file("data15x15_rotated24_image_id.json")
    max_id = json_data_ids[-1]["data"][-1][0]
    print(f"Max id: {max_id}")

    json_index = 0
    j_index = 0

    batch_size = 100
    image_array = []
    print(len(json_data_ids) * 24)

    for i in range(0, max_id + 1):
        image = Image.open(image_path + f"/image{i + 1}.jpeg").convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        embedding = None
        with torch.no_grad():
            embedding = model(image).squeeze().cpu()

        data_index = i % 24
        json_index = i // 24
        json_data_ids[json_index]["data"][data_index] = embedding.tolist()

        if i % batch_size == 0:
            print(f"Batch {i // batch_size} processed")

    write_other_data_to_file("data15x15_rotated24_image_embeddings.json", json_data_ids)


if __name__ == "__main__":
    base_path = './data/dataset15x15'
    embedding = build_resnet18_embeddings(base_path)
