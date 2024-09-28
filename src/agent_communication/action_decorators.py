from functools import wraps
import time

from src.agent_communication.communication_controller import wait_for_response_event, clear_response_event


def wait_agent_response(action):
    """
    Waits for agent response after sending an action.
    """

    @wraps(action)
    def decorator(*args):
        action(*args)
        wait_for_response_event()
        clear_response_event()
        # time.sleep(1)

    return decorator


def sleepy(sleep_time: float):
    def wrapper(action):
        """
        Sleeps for a while before executing the function
        """

        @wraps(action)
        def decorator(*args):
            time.sleep(sleep_time)
            action(*args)

        return decorator

    return wrapper
