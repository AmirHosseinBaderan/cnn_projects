import torch.nn as nn


class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 32x128
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # 16x64
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # 8x32
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # 4x32
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # 2x32
            nn.Conv2d(512, 512, kernel_size=(2, 1)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)