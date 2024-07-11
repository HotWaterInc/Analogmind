from src.configs_setup import configs
from src.modules.external_communication import start_server
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror


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

    # run_variational_autoencoder()
