def get_closest_datapoint_to_xy(self, target_x, target_y) -> str:
    def get_datapoint_metadata_coords(self, name):
        """
        Returns the metadata coordinates of a datapoint
        """
        return [self.raw_env_data_map[name]["params"]["x"], self.raw_env_data_map[name]["params"]["y"]]

    """
    Returns the closest datapoint to a certain x, y coordinate
    """
    closest_datapoint = None
    closest_distance = 100

    for item in self.raw_env_data:
        x, y = item["params"]["x"], item["params"]["y"]
        distance = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_datapoint = item["name"]

    return closest_datapoint
