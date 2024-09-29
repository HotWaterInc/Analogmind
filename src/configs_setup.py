from src.agent_communication.configs import config_communication_controller, config_response_controller


def config_simulation_communication(agent_responded_trigger):
    config_communication_controller()
    config_response_controller(agent_responded_trigger)
