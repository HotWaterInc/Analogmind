def sample_n_random_datapoints(self, sample_size: int) -> List[str]:
    """
    Samples a number of random datapoints
    """
    if sample_size > len(self.raw_env_data):
        warnings.warn("Sample size is larger than the number of datapoints, returning all datapoints")
        sample_size = len(self.raw_env_data)

    return np.random.choice([item["name"] for item in self.raw_env_data], sample_size, replace=False)


def sample_adjacent_datapoints_connections_raw_data(self, sample_size: int) -> List[RawConnectionData]:
    """
    Samples a number the adjacent datapoints

    :param sample_size: the number of datapoints to sample
    """
    return np.random.choice(self.raw_connections_data, sample_size, replace=False)
