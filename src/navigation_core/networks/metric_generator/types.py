from torch.utils.data import Dataset


class WalkData(Dataset):
    def __init__(self, start, end, distance):
        self.starts = start
        self.ends = end
        self.distances = distance

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        return self.starts[idx], self.ends[idx], self.distances[idx]
