""" Custom Transform Dataset Class for PyTorch """
import os
from PIL import Image
from torch.utils.data import Dataset

class TransformDataset(Dataset):
    def __init__(self, base_dataset, image_transform=None, gt_transform=None):
        self.base_dataset = base_dataset
        self.image_transform = image_transform
        self.gt_transform = gt_transform

        # both None or both different than None
        if image_transform == None and gt_transform != None:
            raise ValueError("Invalid transforms!")
        if image_transform != None and gt_transform == None:
            raise ValueError("Invalid transforms!")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # get image and groundtruth
        image, ground_truth = self.base_dataset[idx]
        # apply image transform
        if self.image_transform:
            image = self.image_transform(image)
        # apply groundtruth transform
        if self.gt_transform:
            ground_truth = self.gt_transform(ground_truth)

        return image, ground_truth
