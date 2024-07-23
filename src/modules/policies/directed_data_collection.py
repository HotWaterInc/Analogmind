import time
import math
from typing import Dict, TypedDict, Generator, List
from src.action_ai_controller import ActionAIController
from src.global_data_buffer import GlobalDataBuffer, empty_global_data_buffer
from src.modules.save_load_handlers.data_handle import write_other_data_to_file

from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute
import threading
from pynput import keyboard
from src.action_robot_controller import detach_robot_sample_distance, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute, \
    detach_robot_rotate_continuous_absolute, detach_robot_forward_continuous, detach_robot_A, detach_robot_D, \
    detach_robot_S, detach_robot_W

last_pressed = None


def on_press(key):
    global last_pressed
    last_pressed = key


def on_release(key):
    global last_pressed
    last_pressed = None

    if key == keyboard.Key.esc:
        return False


def listener_thread():
    keyboard.Listener(on_press=on_press, on_release=on_release).start()
    # with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    #     pass
    #     listener.join()


def directed_data_collection() -> Generator[None, None, None]:
    """
    WASD Controlled data collection
    """

    # Create and start the listener
    global last_pressed
    listener_thread()

    valid_keys_map = {
        "w": keyboard.KeyCode.from_char('w'),
        "s": keyboard.KeyCode.from_char('s'),
        "a": keyboard.KeyCode.from_char('a'),
        "d": keyboard.KeyCode.from_char('d')
    }
    valid_keys = list(valid_keys_map.values())

    while True:
        time.sleep(0.05)
        if last_pressed is not None and last_pressed in valid_keys:
            if last_pressed == valid_keys_map["w"]:
                detach_robot_W()
            elif last_pressed == valid_keys_map["s"]:
                detach_robot_S()
            elif last_pressed == valid_keys_map["a"]:
                detach_robot_A()
            elif last_pressed == valid_keys_map["d"]:
                detach_robot_D()
            elif last_pressed == keyboard.Key.esc:
                break
            # last_pressed = None
            yield
