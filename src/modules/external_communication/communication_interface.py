import threading
from typing import Callable


class CommunicationInterface:
    __instance = None

    send_data: Callable[[any], any] = None
    receive_data: Callable[[any], any] = None
    start_server: Callable[[any], any] = None
    server_started: bool = False

    send_data_queue = []

    @staticmethod
    def get_instance():
        if CommunicationInterface.__instance is None:
            CommunicationInterface()
        return CommunicationInterface.__instance

    def __init__(self):
        if CommunicationInterface.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            CommunicationInterface.__instance = self


def send_data(json_data):
    # abstract send_data function easy to use
    communication = CommunicationInterface.get_instance()

    print("server value in sendata: ", communication.server_started)
    if communication.server_started:
        print("sending in started")
        communication.send_data(json_data)
        return 0
    else:
        print("putting in queue")
        # if you call send_data before the server thread started, queue the data
        communication.send_data_queue.append(json_data)


def send_pending_data():
    communication = CommunicationInterface.get_instance()
    print("in pending data", communication)
    for data in communication.send_data_queue:
        send_data(data)


def receive_data(data):
    communication = CommunicationInterface.get_instance()
    communication.receive_data(data)


def set_server_started():
    communication = CommunicationInterface.get_instance()
    communication.server_started = True
    print("ADDRESS", communication)

    print("Server started !!!")
    print("value after start: ", communication.server_started)

    # Use a thread because this callback already runs in the thread that started the server ( since the callback is called by the server )
    # and we don't want to block the server, or use asyncio.run again in the callback since it does not work
    thread = threading.Thread(target=send_pending_data)
    thread.start()


def start_server():
    communication = CommunicationInterface.get_instance()
    if communication.start_server is not None:
        communication.start_server(set_server_started)
