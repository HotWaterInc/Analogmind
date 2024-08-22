from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.variants.exploration.others.neighborhood_network import NeighborhoodDistanceNetwork
from typing import List


def fill_augmented_connections_distances(additional_connections: List[any], storage: StorageSuperset2,
                                         neighborhood_network: NeighborhoodDistanceNetwork):
    additional_connections_augmented = []
    for connection in additional_connections:
        start = connection["start"]
        end = connection["end"]
        start_data = storage.get_datapoint_data_tensor_by_name(start)[0]
        end_data = storage.get_datapoint_data_tensor_by_name(end)[0]
        distance = neighborhood_network(start_data, end_data).squeeze()

        additional_connections_augmented.append({
            "start": start,
            "end": end,
            "distance": distance,
            "direction": None
        })

    return additional_connections_augmented
