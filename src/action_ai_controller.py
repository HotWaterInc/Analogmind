import time
from typing import Dict
from src.global_data_buffer import GlobalDataBuffer


class ActionAIController:
    """
    Simple set of action functions to be used by the AI to control the robot.
    Think of them as a simple API, or assembly instruction
    """
    __instance = None

    callback = None

    @staticmethod
    def get_instance():
        if ActionAIController.__instance is None:
            ActionAIController()
        return ActionAIController.__instance

    def __init__(self):
        if ActionAIController.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ActionAIController.__instance = self


def handle_json_propagation(json_data: Dict):
    """
    Handles "emiting" json data from the robot to the rest of the application
    """
    action_ai_controller: ActionAIController = ActionAIController.get_instance()

    if json_data.get("data") is not None:
        # This was a data response from a sample action
        # TODO: Needs refactor, looks horrible
        data_buffer: GlobalDataBuffer = GlobalDataBuffer.get_instance()
        data_buffer.buffer = json_data
        data_buffer.empty_buffer = False


def detach_action(json_data):
    action_ai_controller = ActionAIController.get_instance()

    handle_json_propagation(json_data)

    if action_ai_controller.callback is not None:
        action_ai_controller.callback()
