import torch
import torch.nn as nn


class TripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(
            self,
            anchor,
            positive,
            negative,
    ):
        positive_distance = torch.nn.functional.pairwise_distance(
            anchor,
            positive,
        )

        negative_distance = torch.nn.functional.pairwise_distance(
            anchor,
            negative,
        )

        loss = torch.relu(
            positive_distance
            -
            negative_distance
            +
            self.margin
        )

        return loss.mean()
