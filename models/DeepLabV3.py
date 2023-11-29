""" Wrapper class for deeplabv3_resnet101 by PyTorch. 
    Model can be found: https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html"""
import torch.nn as nn
import torchvision.models.segmentation as models


class ResNet101(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        # load the model from PyTorch torchvision
        self.model = models.deeplabv3_resnet101(num_classes=1, weights=weights)

    def forward(self, x):
        output = self.model(x)
        output = output["out"]
        return output
