import sys
import time

from modules.external_communication import start_server, CommunicationInterface
from src.configs_setup import get_instance_config
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror
from src.modules.external_communication.communication_interface import send_data, CommunicationInterface
from src.utils import get_instance


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def send_teleport_json(x, y):
    data = {
        "action_type": "TELEPORT_TO",
        "args": {
            "x": x,
            "y": y
        }
    }
    send_data(data)


def take_user_input():
    while True:
        x = input("Enter x: ")
        y = 1
        print("ADDRESS IN MAIN", CommunicationInterface.get_instance())
        send_teleport_json(x, y)


def declare_communication():
    print("ADDRESS IN MAIN", CommunicationInterface.get_instance())
    print("ADDR MODULE", CommunicationInterface)


from modules.external_communication.communication_interface import CommunicationInterface
from src.modules.external_communication.communication_interface import \
    CommunicationInterface as CommunicationInterface2

if __name__ == "__main__":
    """
    This entry is used to develop and test while ai models are still training in the background.
    """

    # configs()

    print(sys.path)
    print(CommunicationInterface.get_instance())
    print(CommunicationInterface2.get_instance())

    print("ADDRESS IN MAIN", CommunicationInterface.get_instance())
    print("ADDRESS IN UTILS", get_instance())
    print("ADDRESS IN CONFIG GETINST", get_instance_config())

    # print("before server thread", CommunicationInterface.get_instance())
    # server_thread = threading.Thread(target=start_server)
    # server_thread.start()
    # print("after server thread", CommunicationInterface.get_instance())
    take_user_input()

    # server_thread.join()
