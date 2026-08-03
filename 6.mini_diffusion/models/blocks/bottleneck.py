import torch.nn as nn

from models.blocks.conv_block import ConvBlock


class Bottleneck(nn.Module):

    def __init__(
        self,
        channels,
        condition_dim=64,
    ):
        super().__init__()


        self.block = ConvBlock(
            channels,
            channels * 2,
        )


        self.condition_projection = nn.Linear(
            condition_dim,
            channels,
        )


    def forward(
        self,
        x,
        condition,
    ):

        condition = self.condition_projection(
            condition
        )


        condition = condition[
            :,
            :,
            None,
            None
        ]


        x = x + condition


        return self.block(x)