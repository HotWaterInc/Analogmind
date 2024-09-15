from typing import Dict


class AgentResponseDataBuffer:
    """
    Simple way of sharing data between different parts of the application
    Works for only simple stuff, but right now it's enough
    """
    __instance = None

    buffer: any = None
    empty_buffer: bool = True

    @staticmethod
    def get_instance():
        if AgentResponseDataBuffer.__instance is None:
            AgentResponseDataBuffer()
        return AgentResponseDataBuffer.__instance

    def __init__(self):
        if AgentResponseDataBuffer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            AgentResponseDataBuffer.__instance = self


def response_data_empty_buffer():
    global_data_buffer: AgentResponseDataBuffer = AgentResponseDataBuffer.get_instance()

    global_data_buffer.buffer = None
    global_data_buffer.empty_buffer = True


def response_data_fill_buffer(json_data: Dict):
    data_buffer: AgentResponseDataBuffer = AgentResponseDataBuffer.get_instance()

    data_buffer.buffer = json_data
    data_buffer.empty_buffer = False
