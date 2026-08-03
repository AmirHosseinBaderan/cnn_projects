import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBlock


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()


        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )


        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
        )


    def forward(
        self,
        x,
        skip,
    ):

        x = self.up(x)


        # برای جلوگیری از اختلاف سایز H,W
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="nearest",
            )


        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )


        x = self.conv(x)

        return x