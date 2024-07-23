import time
import math
from typing import Dict, TypedDict, Generator, List
from src.action_ai_controller import ActionAIController
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.save_load_handlers.data_handle import write_other_data_to_file

from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute, \
    detach_robot_sample_image
import threading


def check_thread():
    current_thread = threading.current_thread()
    print(f"Function is running on thread: {current_thread.name} (ID: {current_thread.ident})")


sample_types = ["distance", "image"]


def grid_data_collection(width: float, height: float, grid_size: int, center_x: float, center_y: float,
                         rotations: int, type: str) -> Generator[None, None, None]:
    """
    Collects data from the environment in a grid pattern with height and width as the boundaries.
    The height and width are split into grid_size x grid_size grid.

    rotations tells in how many parts are the 360 degrees split into.

    Works by having a yield after every action send, which will be triggered by the response from the robot
    """
    if type not in sample_types:
        raise Exception(f"Invalid sample type {type}")

    # we do -1 because the grid size also includes starting ending points
    # if we divide by grid_size, in the loop below you will only traverse up to grid_size-1 which does not include the last point
    # max is used to prevent division by 0 ( if grid size is 1 we have only 1 position anyway so it doesn't matter )
    step_size_x = width / max((grid_size - 1), 1)
    step_size_y = height / max((grid_size - 1), 1)

    # Calculate the starting x and y positions
    start_x = center_x - width / 2
    start_y = center_y - height / 2

    # make sure to specify this one in radians not degrees
    # here we don't use -1 because the last position is the same as the initial one ( 0 rad and 2*pi rad )
    rotation_step = 2 * math.pi / rotations
    actionAiController: ActionAIController = ActionAIController.get_instance()

    all_data: List[Dict] = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the x and y positions
            x = start_x + i * step_size_x
            y = start_y + j * step_size_y

            detach_robot_teleport_absolute(x, y)
            yield

            current_datapoint = {}
            current_name = f"{i}_{j}"
            data_arr = []
            print("COLLECTING DATA FOR", current_name)

            for k in range(rotations):
                # Calculate the angle
                angle = k * rotation_step
                print("ROTATING TO", angle)

                detach_robot_rotate_absolute(angle)
                yield

                if type == "image":
                    detach_robot_sample_image()
                elif type == "distance":
                    detach_robot_sample_distance()
                yield
                # here data buffer should be filled

                global_data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
                buffer = global_data_buffer.buffer
                data_arr.append(buffer["data"])
                empty_global_data_buffer()

            current_datapoint["name"] = current_name
            current_datapoint["data"] = data_arr
            print(data_arr)
            current_datapoint["params"] = {
                "x": x,
                "y": y,
                "i": i,
                "j": j
            }

            all_data.append(current_datapoint)

    print("POLICY HAS FINISHED, SAVING DATA")

    random_hash = str(time.time())
    write_other_data_to_file(f"data{grid_size}x{grid_size}_rotated{rotations}_{random_hash}.json", all_data)
    yield
