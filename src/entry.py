from src.autoencoder import run_autoencoder
from configs_init import configs
from modules.external_communication import start_server, send_data
import threading
from src.modules.data_handlers.ai_models_handle import save_ai, load_latest_ai, load_manually_saved_ai, AIType

from autoencoder import *

if __name__ == "__main__":
    configs()

    # server_thread = threading.Thread(target=start_server, daemon=True)
    # server_thread.start()
    # server_thread.join()

    # autoencoder = load_manually_saved_ai("autoenc_8x8.pth")
    # run_autoencoder()
    pass
