""" Custom Dataset Class for PyTorch """
import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_folder, gt_folder, transform=None):
        self.image_folder = image_folder
        self.gt_folder = gt_folder
        self.transform = transform
        
        # list the file names of images and ground truths
        self.image_files = os.listdir(image_folder)
        self.gt_files = os.listdir(gt_folder)

        # ensure that the two lists have the same length
        assert len(self.image_files) == len(self.gt_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        gt_path = os.path.join(self.gt_folder, self.gt_files[idx])

        image = Image.open(image_path)
        ground_truth = Image.open(gt_path)

        if self.transform:
            image = self.transform(image)
            ground_truth = self.transform(ground_truth)

        return image, ground_truth
