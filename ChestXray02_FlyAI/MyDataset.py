import os

from torch.utils.data import Dataset
import torch
from path import DATA_PATH
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    def __init__(self, input_x, input_y, transform=None):
        self.file_names = input_x
        self.label = input_y
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        path = os.path.join(DATA_PATH, self.file_names[idx]['image_path'])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # image = np.transpose(image, (2, 0, 1))
        label = self.label[idx]['labels']

        return image, label
