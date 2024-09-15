from src.modules.agent_communication.communication_controller import wait_for_response_event, clear_response_event


def wait_agent_response(action):
    """
    Waits for agent response after sending an action.
    """

    def decorator(*args):
        action(*args)
        wait_for_response_event()
        clear_response_event()
        # time.sleep(1)

    return decorator
