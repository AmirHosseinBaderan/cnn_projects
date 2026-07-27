import torch.nn as nn


class DetectionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.box_loss_fn = nn.SmoothL1Loss()
        self.objectness_loss_fn = nn.BCEWithLogitsLoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        prediction,
        target,
    ):
        box_loss = self.compute_box_loss(
            prediction,
            target,
        )

        objectness_loss = self.compute_objectness_loss(
            prediction,
            target,
        )

        classification_loss = self.compute_classification_loss(
            prediction,
            target,
        )

        total_loss = (
            box_loss
            + objectness_loss
            + classification_loss
        )

        return {
            "loss": total_loss,
            "box_loss": box_loss,
            "objectness_loss": objectness_loss,
            "classification_loss": classification_loss,
        }

    def compute_box_loss(
        self,
        prediction,
        target,
    ):
        raise NotImplementedError

    def compute_objectness_loss(
        self,
        prediction,
        target,
    ):
        raise NotImplementedError

    def compute_classification_loss(
        self,
        prediction,
        target,
    ):
        raise NotImplementedError