def tanh_all_data(self):
    # normalizes all the data
    data = self.nodes_get_data()
    normalized_data = np.tanh(np.array(data))

    length = len(self.raw_env_data)
    for i in range(length):
        self.raw_env_data[i]["data"] = normalized_data[i]
        name = self.raw_env_data[i]["name"]
        self.raw_env_data_map[name]["data"] = normalized_data[i]


def normalize_all_data(self):
    # normalizes all the data
    data = self.nodes_get_data()
    normalized_data = normalize_data_min_max(np.array(data))

    length = len(self.raw_env_data)
    for i in range(length):
        self.raw_env_data[i]["data"] = normalized_data[i]
        name = self.raw_env_data[i]["name"]
        self.raw_env_data_map[name]["data"] = normalized_data[i]
