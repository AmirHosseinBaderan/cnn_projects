import torch
import torch.nn as nn

from models.modules.conv import ConvBlock


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.stage1 = ConvBlock(3, 32, stride=2)

        self.stage2 = ConvBlock(32, 64, stride=2)

        self.stage3 = ConvBlock(64, 128, stride=2)

        self.stage4 = ConvBlock(128, 256, stride=2)

        self.stage5 = ConvBlock(256, 512, stride=2)

    def forward(
        self,
        x: torch.Tensor,
    ):

        x = self.stage1(x)
        x = self.stage2(x)

        p3 = self.stage3(x)

        p4 = self.stage4(p3)

        p5 = self.stage5(p4)

        return p3, p4, p5