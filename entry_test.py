from src.configs_setup import configs_communication, config_data_collection_pipeline

from src.modules.external_communication import start_server
import threading
from src.modules.save_load_handlers.data_handle import read_other_data_from_file, write_other_data_to_file

from src.ai.variants.raw_direction_images import policy_navigation_image_rawdirect


def build_resnet18_embeddings():
    # Check if CUDA is available and set the device
    json_data_ids = read_other_data_from_file("data5x5_rotated24_images_id.json")
    new_json_connections = []

    lng = len(json_data_ids)
    for i in range(lng - 1):
        for j in range(i + 1, lng):
            name_start = json_data_ids[i]["name"]
            name_end = json_data_ids[j]["name"]

            i_start = json_data_ids[i]["params"]["i"]
            j_start = json_data_ids[i]["params"]["j"]
            i_end = json_data_ids[j]["params"]["i"]
            j_end = json_data_ids[j]["params"]["j"]

            json_dt = {
                "start": name_start,
                "end": name_end,
                "distance": 1.0,
                "direction": [
                    i_end - i_start,
                    j_end - j_start
                ]
            }

            if abs(i_end - i_start) + abs(j_end - j_start) == 1:
                new_json_connections.append(json_dt)
            elif abs(i_end - i_start) > 1:
                print(f"broken at i_start: {i_start}, i_end: {i_end}")
                break

    write_other_data_to_file("data5x5_connections.json", new_json_connections)


def navigation_direction_raw_image():
    configs_communication()
    generator = navigation_image_rawdirect()

    config_data_collection_pipeline(generator)
    server_thread = threading.Thread(target=start_server, name="ServerThread")
    server_thread.start()

    server_thread.join()


if __name__ == "__main__":
    navigation_direction_raw_image()
    # build_resnet18_embeddings_CORRECT()
    pass
