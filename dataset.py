import json
from torch.utils.data import Dataset
import torch
import os
import utils
import zipfile


class EmbeddingDataset(Dataset):
    def __init__(self, root, transform=None, normalize=True):
        self.data = []
        self.transform = transform
        archive_path = os.path.join(root, 'embeddings-zip.zip')
        archive = zipfile.ZipFile(archive_path, 'r')
        for name in archive.namelist():
            with archive.open(name, 'r') as file:
                embedding = json.load(file)
                self.data.append(embedding)
        if normalize:
            self.data = utils.normalize_dataset(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data[idx]
        if self.transform:
            embedding = self.transform(embedding)
        return embedding
