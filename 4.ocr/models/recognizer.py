import torch
import torch.nn as nn
from sympy import sequence

from models.modules.cnn import CNNFeatureExtractor
from models.modules.sequence import (
    SequenceConverter,
    BidirectionalLSTM
)

from models.modules.classifier import CTCClassifier


class CRNN(nn.Module):
    def __init__(
            self,
            num_classes,
            hidden_size=256,
    ):
        super().__init__()

        self.cnn = CNNFeatureExtractor()
        self.converter = SequenceConverter()
        self.sequence = BidirectionalLSTM(
            input_size=512,
            hidden_size=hidden_size,
        )

        self.classifier = CTCClassifier(
            input_size=hidden_size * 2,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.converter(x)
        x = self.sequence(x)

        logits = self.classifier(x)
        sequence_length = logits.size(1)

        batch_size = logits.size(0)

        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=sequence_length,
            dtype=torch.long,
            device=logits.device
        )

        return {
            "logits": logits,
            "input_lengths": input_lengths
        }