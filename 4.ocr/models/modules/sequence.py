import torch
import torch.nn as nn


class SequenceConverter(nn.Module):

    def forward(self, x):

        # x : (B, C, H, W)

        assert x.size(2) == 1, \
            f"Expected feature height = 1, got {x.size(2)}"

        x = x.squeeze(2)

        # (B, C, W)

        x = x.permute(0, 2, 1)

        # (B, W, C)

        return x

class BidirectionalLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return output