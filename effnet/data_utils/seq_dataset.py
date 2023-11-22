import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, feature_path, idx):
        self.data_instances = []
        features = torch.load(feature_path)
        for feature in features:
            self.data_instances.append((feature[0], idx))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)
