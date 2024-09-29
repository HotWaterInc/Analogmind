from src.agent_communication import CommunicationController
from src.agent_communication.implementations.websocket_server import send_data_websockets, start_websockets
from src.agent_communication.response_controller import AgentResponseController, receive_response


def config_communication_controller() -> None:
    """
    Configures the communication controller with whatever form of communication is desired
    """
    # abstracts away the communication details
    communication: CommunicationController = CommunicationController.get_instance()

    communication.start_server = start_websockets
    communication.send_data = send_data_websockets
    communication.receive_response = receive_response

    communication.server_started_callbacks.append(lambda: print("Connection established with robot"))


def config_response_controller(agent_responded) -> None:
    """
    Essentially configures the communication loop between AI and agent, synchronizing the communication and AI threads
    Ensures AI waits for server to response before proceeding in its endeavours further

    So: AI sends action, AI stalls, agent sends back feedback, AI receives back and continues
    """
    response_controller: AgentResponseController = AgentResponseController.get_instance()
    response_controller.callbacks.append(lambda: agent_responded())
