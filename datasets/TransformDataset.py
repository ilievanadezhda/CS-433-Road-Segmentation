""" Custom Transform Dataset Class for PyTorch """
from torch.utils.data import Dataset


class TransformDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        random_transform=None,
        image_transform=None,
        gt_transform=None,
    ):
        self.base_dataset = base_dataset
        self.random_transform = random_transform
        self.image_transform = image_transform
        self.gt_transform = gt_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # get image and groundtruth
        image, ground_truth = self.base_dataset[idx]
        # make a dictionary
        sample = {"image": image, "ground_truth": ground_truth}
        # apply random transform to the sample (applies to both image and groundtruth)
        if self.random_transform:
            sample = self.random_transform(sample)
        # get image and groundtruth
        image, ground_truth = sample["image"], sample["ground_truth"]
        # apply image transform
        if self.image_transform:
            image = self.image_transform(image)
        # apply groundtruth transform
        if self.gt_transform:
            ground_truth = self.gt_transform(ground_truth)
        # return image and groundtruth
        return image, ground_truth
