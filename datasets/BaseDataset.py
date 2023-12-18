""" Custom Base Dataset Class for PyTorch """
import os
from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, image_folders, gt_folders):
        self.image_folders = image_folders
        self.gt_folders = gt_folders

        # list the file names of images and ground truths
        self.image_files = sorted(
            [
                os.path.join(image_folder, file)
                for image_folder in image_folders
                for file in os.listdir(image_folder)
            ]
        )
        self.gt_files = sorted(
            [
                os.path.join(gt_folder, file)
                for gt_folder in gt_folders
                for file in os.listdir(gt_folder)
            ]
        )

        # ensure that the two lists have the same length
        assert len(self.image_files) == len(self.gt_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        ground_truth = Image.open(self.gt_files[idx])
        # convert the floats of the ground truth to integers, threshold 10
        ground_truth = ground_truth.point(lambda x: 0 if x < 10 else 255)

        return image, ground_truth
