# class ActionAIController:
#     """
#     Simple set of action functions to be used by the AI to control the robot.
#     Think of them as a simple API, or assembly instruction
#     """
#     __instance = None
#
#     action1 = None
#     action2 = None
#     action3 = None
#
#     @staticmethod
#     def get_instance():
#         if ActionController.__instance is None:
#             ActionController()
#         return ActionController.__instance
#
#     def __init__(self):
#         if ActionController.__instance is not None:
#             raise Exception("This class is a singleton!")
#         else:
#             ActionController.__instance = self
#

def detach_action(json_data):
    print("Detaching action")
    # actions = ActionController.get_instance()
    print(json_data)