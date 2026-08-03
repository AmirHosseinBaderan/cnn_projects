import torch
import torch.nn as nn


class ConditionProjection(nn.Module):

    def __init__(
        self,
        embedding_dim,
        channels,
    ):
        super().__init__()


        self.projection = nn.Linear(
            embedding_dim,
            channels,
        )


    def forward(
        self,
        condition,
    ):

        return self.projection(condition)