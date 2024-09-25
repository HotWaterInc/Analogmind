from torch import nn

from src.runtime_storages.storage_struct import StorageStruct
from src import runtime_storages as storage
from src.runtime_storages.types import NodeAuthenticData, ConnectionNullData, ConnectionSyntheticData, \
    ConnectionAuthenticData
from src.save_load_handlers.data_handle import read_other_data_from_file


def load_storage_with_base_data(storage_struct: StorageStruct, datapoints_filename: str,
                                connections_filename: str) -> any:
    random_walk_datapoints = read_other_data_from_file(datapoints_filename)
    random_walk_connections = read_other_data_from_file(connections_filename)
    nodes = []
    connections_authentic = []
    connections_synthetic = []
    connections_null = []
    for element in random_walk_datapoints:
        node = NodeAuthenticData(
            name=element["name"],
            datapoints_array=element["data"],
            params=element["params"]
        )
        nodes.append(node)

    cnt_null = 0
    cnt_synth = 0
    cnt_auth = 0
    for element in random_walk_connections:
        if element["end"] == None:
            cnt_null += 1
            conn = ConnectionNullData(
                name=element["start"],
                start=element["start"],
                distance=element["distance"],
                direction=element["direction"]
            )
            connections_null.append(conn)
        elif element["markings"]["distance"] == "synthetic":
            cnt_synth += 1
            conn = ConnectionSyntheticData(
                name=element["start"],
                start=element["start"],
                end=element["end"],
                distance=element["distance"],
                direction=element["direction"]
            )
            connections_synthetic.append(conn)
        else:
            cnt_auth += 1
            conn = ConnectionAuthenticData(
                name=element["start"],
                start=element["start"],
                end=element["end"],
                distance=element["distance"],
                direction=element["direction"]
            )
            connections_authentic.append(conn)

    print("Connections null", cnt_null)
    print("Connections synthetic", cnt_synth)
    print("Connections authentic", cnt_auth)
    print(len(random_walk_datapoints))
    return storage_struct
