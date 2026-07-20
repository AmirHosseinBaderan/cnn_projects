import math

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            #
            # Input
            # H = 32
            # W = Variable
            #

            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(True),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            ),

            nn.Conv2d(
                256,
                512,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(True),

            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            ),

            nn.Conv2d(
                512,
                512,
                kernel_size=(2, 1),
            ),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.features(x)

    @staticmethod
    def _conv_output_size(
        size,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        return math.floor(
            (
                size
                + (2 * padding)
                - (dilation * (kernel_size - 1))
                - 1
            ) / stride
            + 1
        )

    def get_output_lengths(
        self,
        input_widths: torch.Tensor,
    ) -> torch.Tensor:
        widths = input_widths.clone()

        # Pool 1
        widths = torch.tensor(
            [
                self._conv_output_size(
                    int(w),
                    kernel_size=2,
                    stride=2,
                )
                for w in widths
            ],
            device=input_widths.device,
        )

        # Pool 2
        widths = torch.tensor(
            [
                self._conv_output_size(
                    int(w),
                    kernel_size=2,
                    stride=2,
                )
                for w in widths
            ],
            device=input_widths.device,
        )

        return widths.long()