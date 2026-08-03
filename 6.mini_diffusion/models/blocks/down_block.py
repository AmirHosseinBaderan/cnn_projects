import torch.nn as nn

from models.blocks.conv_block import ConvBlock


class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()


        self.block = ConvBlock(
            in_channels,
            out_channels,
        )


        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )


    def forward(self, x):

        x = self.block(x)

        skip = x

        x = self.downsample(x)

        return x, skip