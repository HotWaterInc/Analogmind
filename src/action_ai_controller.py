import time


class ActionAIController:
    """
    Simple set of action functions to be used by the AI to control the robot.
    Think of them as a simple API, or assembly instruction
    """
    __instance = None

    received_data: bool = False
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


def detach_action(json_data):
    action_ai_controller = ActionAIController.get_instance()
    action_ai_controller.received_data = True
    if action_ai_controller.callback is not None:
        print("detaching action + waiting")
        time.sleep(1)
        print("actually detaching")
        action_ai_controller.callback()
