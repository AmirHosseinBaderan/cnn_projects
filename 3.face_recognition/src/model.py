import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceEmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                128,
                embedding_dim
            ),
        )

    def forward(self,x):
        x = self.features(x)
        x = self.embedding(x)
        x = F.normalize(
            x,
            p=2,
            dim=1
        )
        return x