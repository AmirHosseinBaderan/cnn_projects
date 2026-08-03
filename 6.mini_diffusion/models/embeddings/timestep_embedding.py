import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim


    def forward(
        self,
        timesteps,
    ):

        half_dim = self.embedding_dim // 2

        factor = math.log(10000) / (half_dim - 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * -factor
        )

        embeddings = (
            timesteps.float()[:, None]
            *
            frequencies[None, :]
        )

        embeddings = torch.cat(
            (
                torch.sin(embeddings),
                torch.cos(embeddings),
            ),
            dim=1,
        )

        return embeddings