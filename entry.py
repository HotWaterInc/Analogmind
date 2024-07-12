from src.configs_setup import configs
from src.modules.external_communication import start_server
import threading
from src.modules.visualizations import run_visualization
from src.ai.models.autoencoder import *
from src.ai.models.variational_autoencoder import *
from src.utils import perror

if __name__ == "__main__":
    configs()

    # start_server_thread()

    run_autoencoder()
    # run_visualization()

    # run_variational_autoencoder()
