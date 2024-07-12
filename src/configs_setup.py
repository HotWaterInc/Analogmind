from src.modules.external_communication import start_websockets
from src.modules.external_communication.websocket_server import send_data_websockets, start_websockets
from src.modules.external_communication.communication_interface import CommunicationInterface, send_pending_data
from src.action_ai_controller import detach_action
from src.action_ai_controller import ActionAIController
from typing import Generator


def configs_communication() -> None:
    """
    Configures the communication interface (basically plugging in the websocket server into the abstract functions)
    """
    # abstracts away the communication details
    communication: CommunicationInterface = CommunicationInterface.get_instance()

    communication.start_server = start_websockets
    communication.send_data = send_data_websockets
    communication.receive_data = detach_action

    communication.server_started_callbacks.append(lambda: print("Connection established with robot"))
    communication.server_started_callbacks.append(send_pending_data)


def config_data_collection_pipeline(policy_generator) -> None:
    """
    Configures the feedback loop between the policy which sends actions to collect data and the retrieved data which
    triggers the policy to send more actions

    So: Policy sends action, policy stalls, robot sends back feedback, policy receives back and continues
    """
    actionai_controller: ActionAIController = ActionAIController.get_instance()
    actionai_controller.callback = lambda: next(policy_generator)

    communication: CommunicationInterface = CommunicationInterface.get_instance()
    communication.server_started_callbacks.append(lambda: print("server started callback, turning on policy generator"))
    communication.server_started_callbacks.append(lambda: next(policy_generator))


def configs() -> None:
    """
    Used to configure various parameters of the project such as communication, purpose,
    This is the dirty hardcoded entry point for the project with all the concrete settings
    """
    configs_communication()
