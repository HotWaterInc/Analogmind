from src.modules.external_communication import start_server
from src.configs_setup import configs_communication, config_data_collection_pipeline
import threading
from src.modules.policies.data_collection import grid_data_collection
from src.modules.policies.navigation8x8_v1_distance import navigation8x8
from src.modules.policies.directed_data_collection import directed_data_collection


def thread1():
    print("thread1")


def start_server_thread():
    thread1 = threading.Thread(target=start_server)
    thread1.start()


if __name__ == "__main__":
    # directed_data_collection_pipeline()
    # navigation8x8pipeline()
    # data_collection_pipeline()

    # run_autoencoder()
    # run_permutor()

    # run_visualization()
    # run_permuted_autoencoder2()
    # run_direction_network2()
    pass
