from configs_init import configs
from modules.external_communication import start_server
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


def testfunc():
    print("Test function called")


if __name__ == "__main__":
    configs()

    # start_server_thread()

    run_autoencoder()
    # run_visualization()

    pass
