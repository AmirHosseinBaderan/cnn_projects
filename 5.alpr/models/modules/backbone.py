import torch
import torch.nn as nn

from models.modules.conv import ConvBlock


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(3, 32, stride=2),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)