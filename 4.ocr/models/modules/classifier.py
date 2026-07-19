import torch.nn as nn

class CTCClassifier(nn.Module):
    def __init__(
            self,
            input_size,
            num_classes,
    ):
        super().__init__()

        self.fc = nn.Linear(
            input_size,
            num_classes,
        )

    def forward(self, x):
        x = self.fc(x)
        return x