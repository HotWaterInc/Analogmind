from src.agent_communication.response_data_buffer_class import response_data_fill_buffer


class AgentResponseController:
    __instance = None

    callbacks = []

    @staticmethod
    def get_instance():
        if AgentResponseController.__instance is None:
            AgentResponseController()
        return AgentResponseController.__instance

    def __init__(self):
        if AgentResponseController.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            AgentResponseController.__instance = self


def receive_response(json_data):
    agent_response_controller: AgentResponseController = AgentResponseController.get_instance()
    response_data_fill_buffer(json_data)

    for callback in agent_response_controller.callbacks:
        callback()
