""" Wrapper class for PyTorch-UNet (unet_carvana) by Milesial. 
    Original code can be found at: https://github.com/milesial/Pytorch-UNet"""
import torch
import torch.nn as nn


class UNetV1(nn.Module):
    def __init__(self, pretrained=False, scale=0.5):
        super().__init__()
        # Load the model from Torch Hub
        self.model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=pretrained, scale=scale)

    def forward(self, x):
        output = self.model(x)
        return output[:, [0], :, :]
