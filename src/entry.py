from src.autoencoder import run_autoencoder
from configs_init import configs
from modules.external_communication import start_server, send_data
import threading

if __name__ == "__main__":
    configs()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # run_autoencoder('')
    send_data("BEfore start from ai")
    server_thread.join()
