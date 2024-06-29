from src.autoencoder import run_autoencoder
from configs_init import configs
from modules.external_communication import start_server, send_data
import threading
from src.modules.data_handlers.ai_models_handle import save_ai, load_latest_ai, load_manually_saved_ai, AIType

from autoencoder import *

def start_server_thread():
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    server_thread.join()


if __name__ == "__main__":
    configs()

    # start_server_thread()
    # run_autoencoder()

    pass
