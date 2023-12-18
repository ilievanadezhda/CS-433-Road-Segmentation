import numpy as np
import torch
import random


def set_seeds(seed=42):
    """Set seeds for reproducibility.

    Args:
        seed (int, optional): Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def step_loader(loader, n_steps=-1):
    """Step loader for training. This is used to create infinite loader for training. We use n_steps instead of n_epochs as metric.

    Args:
        loader (DataLoader): data loader for training
        n_steps (int, optional): number of steps to train. Defaults to -1.

    Yields:
        step: current step
        batch: current batch
    """
    step = 0
    while True:
        for batch in loader:
            yield step, batch
            step += 1
            if step == n_steps:
                return


def calculate_metrics(preds, labels, threshold=0.5):
    """Calculate metrics for evaluation.

    Args:
        preds: predicted labels
        labels: groundtruth labels
        threshold (float, optional): threshold for prediction. Defaults to 0.5.

    Returns:
        pixel_accuracy: pixel accuracy
        iou: intersection-over-union
        f1_score: F1 score
    """
    smooth = 1e-6
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    # Pixel Accuracy
    pixel_accuracy = (preds == labels).sum().item() / (labels.numel() + smooth)

    # Intersection-Over-Union (IoU)
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    # Precision and Recall for F1 Score
    true_positives = (preds * labels).sum()
    predicted_positives = preds.sum()
    actual_positives = labels.sum()

    precision = true_positives / (predicted_positives + smooth)
    recall = true_positives / (actual_positives + smooth)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + smooth)

    return pixel_accuracy, iou.item(), f1_score.item()


def batch_mean_and_sd(loader):
    """Calculates the mean and standard deviation of a dataset.

    Args:
        loader: dataset loader

    Returns:
        mean: mean of dataset
        std: standard deviation of dataset
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)
    return mean, std
