import torch
import torch.nn as nn


class LabelEmbedding(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_classes,
            embedding_dim,
        )


    def forward(
        self,
        labels,
    ):
        return self.embedding(labels)