""" Wrapper class for deeplabv3_resnet101 by PyTorch. 
    Model can be found: 
    https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html#torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
"""
import torch.nn as nn
import torchvision.models.segmentation as models


class ResNet50(nn.Module):
    def __init__(self, weights="DEFAULT"):
        super().__init__()
        # load the model from PyTorch torchvision
        self.model = models.deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(
            256, 1, kernel_size=(1, 1), stride=(1, 1)
        )  # change the number of classes to 1

    def forward(self, x):
        output = self.model(x)
        output = output["out"]
        return output
