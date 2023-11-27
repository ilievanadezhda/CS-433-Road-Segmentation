""" Wrapper class for brain-segmentation-pytorch (unet) by mateuszbuda. 
    Original code can be found at: https://github.com/mateuszbuda/brain-segmentation-pytorch"""
import torch
import torch.nn as nn


class UNetV2(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, init_features=32, pretrained=False
    ):
        super().__init__()
        # Load the model from Torch Hub
        self.model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            pretrained=pretrained,
        )

    def forward(self, x):
        output = self.model(x)
        return output
