import torch
import math


class TimeEmbedding:

    def __init__(
        self,
        embedding_dim: int,
    ):
        self.embedding_dim = embedding_dim


    def __call__(
        self,
        timesteps,
    ):

        half_dim = self.embedding_dim // 2


        embeddings = math.log(10000) / (
            half_dim - 1
        )


        embeddings = torch.exp(
            torch.arange(half_dim)
            *
            -embeddings
        )


        embeddings = (
            timesteps[:, None]
            *
            embeddings[None, :]
        )


        embeddings = torch.cat(
            [
                torch.sin(embeddings),
                torch.cos(embeddings),
            ],
            dim=1
        )


        return embeddings