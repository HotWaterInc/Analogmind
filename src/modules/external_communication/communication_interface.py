import threading
from typing import Callable, Dict, TypedDict, Union, List


class CommunicationInterface:
    __instance = None

    send_data: Callable[[Dict[str, any]], any] = None
    receive_data: Callable[[Dict[str, any]], any] = None
    start_server: Callable[[any], any] = None
    server_started: bool = False

    send_data_queue = []

    server_started_callbacks = []

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


class TeleportAbsoluteAction(TypedDict):
    action_type: str
    x: float
    y: float


class TeleportRelativeAction(TypedDict):
    action_type: str
    dx: float
    dy: float


class RotateAbsoluteAction(TypedDict):
    action_type: str
    angle: float


class RotateAbsoluteContinuousAction(TypedDict):
    action_type: str
    angle: float


class ForwardContinuousAction(TypedDict):
    action_type: str
    distance: float


class RotateRelativeAction(TypedDict):
    action_type: str
    dangle: float


class SampleDistanceAction(TypedDict):
    action_type: str
    pass


class SampleImageAction(TypedDict):
    action_type: str
    pass


class OtherAction(TypedDict):
    action_type: str
    pass


ActionTypeTeleportAbsolute = "teleport_absolute"
ActionTypeTeleportRelative = "teleport_relative"
ActionTypeRotateAbsolute = "rotate_absolute"
ActionTypeRotateRelative = "rotate_relative"

ActionTypeContRotateAbsolute = "cont_rotate_absolute"
ActionTypeContForward = "cont_forward"

ActionTypeSampleDistance = "sample_distance"
ActionTypeSampleImage = "sample_image"

ActionTypeContW = "w"
ActionTypeContA = "a"
ActionTypeContS = "s"
ActionTypeContD = "d"

action_types: List[str] = [
    ActionTypeTeleportAbsolute, ActionTypeTeleportRelative, ActionTypeRotateAbsolute, ActionTypeRotateRelative,
    ActionTypeSampleDistance, ActionTypeContRotateAbsolute, ActionTypeContForward
]

JsonDataAction = Union[
    TeleportAbsoluteAction, TeleportRelativeAction, RotateAbsoluteAction, RotateRelativeAction, SampleImageAction, RotateAbsoluteContinuousAction, ForwardContinuousAction]


def send_data(json_data: JsonDataAction) -> None:
    """
    Global send data function, use this one as a wrapper to CommunicationInterface.get_instance().send_data
    """
    # abstract send_data function easy to use
    # if json_data.get("action_type") is None:
    #     raise Exception("action_type is required in json_data")
    # if json_data["action_type"] not in action_types:
    #     raise Exception(f"action_type {json_data['action_type']} is not a valid action type")

    communication: CommunicationInterface = CommunicationInterface.get_instance()

    if communication.server_started:
        communication.send_data(json_data)
    else:
        # if you call send_data before the server thread started, queue the data
        communication.send_data_queue.append(json_data)


def send_pending_data():
    communication = CommunicationInterface.get_instance()
    for data in communication.send_data_queue:
        send_data(data)


def receive_data(data: Dict[str, any]):
    communication = CommunicationInterface.get_instance()
    communication.receive_data(data)


def set_server_started():
    communication = CommunicationInterface.get_instance()
    communication.server_started = True

    # Use a thread because this callback already runs in the thread that started the server ( since the callback is called by the server )
    # and we don't want to block the server, or use asyncio.run again in the callback since it does not work

    # Update: Fixed the issue in the send_data function by using asyncio.create_task along with .run for the 2 cases
    # where the main thread sends data and the server thread sends data
    for callback in communication.server_started_callbacks:
        callback()


def start_server():
    communication = CommunicationInterface.get_instance()
    if communication.start_server is not None:
        communication.start_server(set_server_started)
