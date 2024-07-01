from modules.external_communication import CommunicationInterface
from modules.external_communication import start_websockets
from modules.external_communication.websockets_server import send_data
from action_controller import detach_action
from action_controller import ActionController


def configs_communication():
    # abstracts away the communication details
    communication = CommunicationInterface.get_instance()

    communication.start_server = start_websockets
    communication.send_data = send_data
    communication.receive_data = detach_action


def config_actions(action1, action2, action3):
    actions = ActionController.get_instance()
    actions.action1 = action1
    actions.action2 = action2
    actions.action3 = action3


def configs():
    """
    Used to configure various parameters of the project such as communication, purpose,
    This is the dirty hardcoded entry point for the project with all the concrete settings
    """
    configs_communication()
