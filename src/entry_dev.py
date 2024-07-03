from configs_init import configs
from modules.external_communication import start_server
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror


def start_server_thread():
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


if __name__ == "__main__":
    """
    This entry is used to develop and test while ai models are still training in the background.
    """
    configs()

    # start_server_thread()

    # run_visualization()

    # run_variational_autoencoder()
