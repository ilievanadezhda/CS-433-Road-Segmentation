""" Wrapper class for brain-segmentation-pytorch (unet) by mateuszbuda. 
    Original code can be found at: https://github.com/mateuszbuda/brain-segmentation-pytorch"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetV3(nn.Module):
    def __init__(
        self,
        encoder="efficientnet-b5",
        encoder_weights="imagenet",
        classes=1,
        activation=None,
    ):
        super().__init__()
        # Load the model from Torch Hub
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )

    def forward(self, x):
        output = self.model(x)
        return output
