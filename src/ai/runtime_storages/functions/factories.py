def _generate_adjacency_data_sample(self, item: RawConnectionData, distance: int = 1) -> AdjacencyDataSample:
    return {
        "start": item["start"],
        "end": item["end"],
        "distance": distance
    }
