import torch.nn as nn


class CTCLossWrapper(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.CTCLoss(
            blank=0,
            reduction="mean",
            zero_infinity=True
        )

    def forward(
        self,
        logits,
        targets,
        input_lengths,
        target_lengths
    ):
        # (B,T,C) -> (T,B,C)
        logits = logits.permute(1, 0, 2)

        return self.loss(
            logits.log_softmax(dim=-1),
            targets,
            input_lengths,
            target_lengths
        )