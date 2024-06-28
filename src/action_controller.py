class ActionController:
    __instance = None

    action1 = None
    action2 = None
    action3 = None

    @staticmethod
    def get_instance():
        if ActionController.__instance is None:
            ActionController()
        return ActionController.__instance

    def __init__(self):
        if ActionController.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ActionController.__instance = self


def detach_action(json_data):
    print("Detaching action")
    actions = ActionController.get_instance()