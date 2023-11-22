import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm


class AudioDataset(Dataset):
    def __init__(self, data_dir, limit=1000000):
        self.data_instances = []
        self.answer_instances = []
        self.pitch_instances = []
        count = 0
        for the_dir in tqdm(os.listdir(data_dir)):
            feature = os.path.join(data_dir, the_dir, "CQT_feature.pt")
            self.data_instances += torch.load(feature)
            count += 1
            if count >= limit:
                print("File limit reached")
                break

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)
