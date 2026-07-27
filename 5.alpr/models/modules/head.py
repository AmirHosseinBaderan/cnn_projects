import torch
import torch.nn as nn

from models.modules.conv import ConvBlock


class DetectionHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.prediction_channels = 5 + num_classes

        self.features = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )

        self.predictor = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.prediction_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.features(x)
        prediction = self.predictor(features)

        return prediction