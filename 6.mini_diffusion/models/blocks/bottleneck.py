import torch.nn as nn

from models.blocks.conv_block import ConvBlock


class Bottleneck(nn.Module):

    def __init__(
        self,
        channels: int,
        condition_dim: int = 64,
    ):
        super().__init__()

        self.condition_projection = nn.Linear(
            condition_dim,
            channels,
        )

        self.block = ConvBlock(
            channels,
            channels * 2,
        )

    def forward(
        self,
        x,
        condition,
    ):

        condition = self.condition_projection(condition)

        condition = condition.unsqueeze(-1).unsqueeze(-1)

        x = x + condition

        x = self.block(x)

        return x