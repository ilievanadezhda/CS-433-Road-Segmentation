""" Wrapper class for unet model from segmentation_models_pytorch library 
    Model can be found: https://segmentation-modelspytorch.readthedocs.io/en/latest/docs/api.html#unet """
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetV3(nn.Module):
    # activation is none since we are using BCEWithLogitsLoss
    def __init__(
        self,
        encoder="resnet50",
        encoder_weights="imagenet",
        classes=1,
        activation=None,
    ):
        super().__init__()
        # load model from segmentation_models_pytorch library
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )

    def forward(self, x):
        output = self.model(x)
        return output
