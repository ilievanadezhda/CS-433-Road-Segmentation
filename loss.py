import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.bcewithlogits = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        dice_loss = 1 - dice

        bce_loss = self.bcewithlogits(y_pred, y_true)

        return dice_loss + bce_loss
