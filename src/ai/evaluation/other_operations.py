import numpy as np
import torch
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage import Storage
from src.utils import array_to_tensor


def get_model_distance_degree1(model: BaseAutoencoderModel, storage: Storage, verbose: bool) -> float:
    """
    Get the average distance between connected pairs (degree 1)
    """
    adjacent_data = storage.get_adjacency_data()
    average_distance = 0

    for connection in adjacent_data:
        start_uid = connection["start"]
        end_uid = connection["end"]

        start_data = storage.get_datapoint_data_tensor_by_name(start_uid)
        end_data = storage.get_datapoint_data_tensor_by_name(end_uid)

        start_embedding = model.encoder_inference(start_data.unsqueeze(0))
        end_embedding = model.encoder_inference(end_data.unsqueeze(0))

        distance_from_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()
        average_distance += distance_from_embeddings

    average_distance /= len(adjacent_data)
    if verbose:
        print(f"Average distance between connected pairs: {average_distance:.4f}")

    return average_distance
