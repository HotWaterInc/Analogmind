from src.modules.external_communication import start_websockets
from src.modules.external_communication.websockets_server import send_data, CommunicationInterface
from action_controller import detach_action
from action_controller import ActionController


def configs_communication() -> None:
    """
    Configures the communication interface (basically plugging in the websocket server into the abstract functions)
    """
    # abstracts away the communication details
    communication = CommunicationInterface.get_instance()
    print("IN CONFIG", communication)

    communication.start_server = start_websockets
    communication.send_data = send_data
    communication.receive_data = detach_action


def config_actions(action1, action2, action3):
    actions = ActionController.get_instance()
    actions.action1 = action1
    actions.action2 = action2
    actions.action3 = action3


def configs() -> None:
    """
    Used to configure various parameters of the project such as communication, purpose,
    This is the dirty hardcoded entry point for the project with all the concrete settings
    """
    configs_communication()
