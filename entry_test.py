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
from src.action_robot_controller import detach_robot_sample, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative

import asyncio


async def function1():
    print("Inside function1")
    function2()


def function2():
    print("Inside function2")
    asyncio.create_task(function3())


async def function3():
    print("Inside function3")


asyncio.run(function1())
