import torch
import torch.nn as nn

from models.modules.cnn import CNNFeatureExtractor
from models.modules.sequence import (
    SequenceConverter,
    BidirectionalLSTM,
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

    def forward(
        self,
        images,
        image_widths=None,
    ):
        # CNN
        features = self.cnn(images)

        # (B,C,1,W) -> (B,W,C)
        sequence = self.converter(features)

        # BiLSTM
        sequence = self.sequence(sequence)

        # Classifier
        logits = self.classifier(sequence)

        batch_size = logits.size(0)
        sequence_length = logits.size(1)

        # CTC input lengths
        if image_widths is None:
            input_lengths = torch.full(
                (batch_size,),
                sequence_length,
                dtype=torch.long,
                device=logits.device,
            )

        else:
            input_lengths = self.cnn.get_output_lengths(
                image_widths.to(logits.device)
            )
            input_lengths = torch.clamp(
                input_lengths,
                max=sequence_length,
            )

        return {
            "logits": logits,
            "input_lengths": input_lengths,
        }