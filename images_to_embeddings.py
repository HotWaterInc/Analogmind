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
