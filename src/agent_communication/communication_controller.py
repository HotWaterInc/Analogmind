import threading
from typing import Callable, Dict, TypedDict, Union, List


class CommunicationController:
    """
    Binding glue between AI and agent communication
    Holds the functions for both seding the data to the server and receiving the response
    Dependency injection is done in configs before the whole workflow starts

    """
    __instance = None

    send_data: Callable[[Dict[str, any]], any] = None
    receive_response: Callable[[Dict[str, any]], any] = None
    start_server: Callable[[any], any] = None
    server_started: bool = False

    send_data_queue = []

    server_started_callbacks = []
    response_event = threading.Event()

    @staticmethod
    def get_instance():
        if CommunicationController.__instance is None:
            CommunicationController()
        return CommunicationController.__instance

    def __init__(self):
        if CommunicationController.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            CommunicationController.__instance = self
            self.initialize_defaults()

    def initialize_defaults(self):
        self.server_started_callbacks.append(_send_pending_data)


def agent_send_data(json_data) -> None:
    """
    Sends data to the agent
    """
    communication: CommunicationController = CommunicationController.get_instance()
    if communication.server_started:
        communication.send_data(json_data)
    else:
        # if you call send_data before the server thread started, queue the data
        communication.send_data_queue.append(json_data)


def _send_pending_data():
    communication = CommunicationController.get_instance()
    for data in communication.send_data_queue:
        agent_send_data(data)


def ai_receive_response(data: Dict[str, any]):
    communication: CommunicationController = CommunicationController.get_instance()
    communication.receive_response(data)


def set_server_started():
    communication = CommunicationController.get_instance()
    communication.server_started = True

    for callback in communication.server_started_callbacks:
        callback()


def start_server():
    communication = CommunicationController.get_instance()
    if communication.start_server is None:
        raise Exception("Server not configured")

    communication.start_server()


def set_response_event():
    communication = CommunicationController.get_instance()
    communication.response_event.set()


def clear_response_event():
    communication = CommunicationController.get_instance()
    communication.response_event.clear()


def wait_for_response_event():
    communication = CommunicationController.get_instance()
    communication.response_event.wait()
