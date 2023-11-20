""" Custom Test Dataset Class for PyTorch """
import os
import re
from PIL import Image
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for folder_name in os.listdir(self.root_dir):
            if folder_name.startswith("test_") and os.path.isdir(os.path.join(self.root_dir, folder_name)):
                folder_path = os.path.join(self.root_dir, folder_name)
                for filename in os.listdir(folder_path):
                    if filename.startswith("test_") and filename.endswith(".png"):
                        image_paths.append(os.path.join(folder_path, filename))
        # sort based on the index in the filename [test_0.png, test_1.png, ...]
        image_paths = sorted(image_paths, key=extract_index)
        # return
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image

def extract_index(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else -1