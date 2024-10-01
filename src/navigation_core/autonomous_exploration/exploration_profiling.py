import time
import math
from typing import List

import numpy as np

from src.agent_communication.action_detach import detach_agent_sample_distances, detach_agent_teleport_relative, \
    detach_agent_rotate_absolute, detach_agent_sample_image, detach_agent_teleport_absolute
from src.navigation_core.autonomous_exploration.common import get_collected_data_distances, random_distance_generator, \
    random_direction_generator, check_direction_validity, get_collected_data_image
from src.runtime_storages.storage_struct import StorageStruct


def exploration_profiling():
    """
    Only collects data, lets the user teleport and rotate the agent.
    """
    detach_agent_teleport_absolute(0, 0)

    exploring = True
    step = 0

    while exploring:
        step += 1
        detach_agent_sample_image()
        image_embedding, angle, coords = get_collected_data_image()
        print(f"Step {step}, coords: {coords}, angle: {angle}")

        time.sleep(1)
        direction = 3 * np.pi / 2
        xy_webots = direction_to_xy_webots(direction, 0.5)
        detach_agent_teleport_relative(xy_webots[0], xy_webots[1])


storage_struct: StorageStruct
