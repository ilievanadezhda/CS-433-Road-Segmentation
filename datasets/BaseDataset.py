""" Custom Base Dataset Class for PyTorch """
import os
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, image_folder, gt_folder):
        self.image_folder = image_folder
        self.gt_folder = gt_folder

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

        return image, ground_truth