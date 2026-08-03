import torch
import torch.nn as nn

from trainer.detection_prediction import DetectionPrediction
from trainer.detection_target import DetectionTarget


class DetectionLoss(nn.Module):

    def __init__(
        self,
        box_weight: float = 5.0,
        objectness_weight: float = 1.0,
        classification_weight: float = 1.0,
    ):
        super().__init__()

        self.box_weight = box_weight
        self.objectness_weight = objectness_weight
        self.classification_weight = classification_weight

        self.box_loss_fn = nn.SmoothL1Loss()
        self.objectness_loss_fn = nn.BCEWithLogitsLoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        prediction: DetectionPrediction,
        target: DetectionTarget,
    ) -> dict[str, torch.Tensor]:

        box_loss = self._box_loss(
            prediction,
            target,
        )

        objectness_loss = self._objectness_loss(
            prediction,
            target,
        )

        classification_loss = self._classification_loss(
            prediction,
            target,
        )

        total_loss = (
            self.box_weight * box_loss
            + self.objectness_weight * objectness_loss
            + self.classification_weight * classification_loss
        )

        return {
            "loss": total_loss,
            "box_loss": box_loss,
            "objectness_loss": objectness_loss,
            "classification_loss": classification_loss,
        }

    def _box_loss(
        self,
        prediction: DetectionPrediction,
        target: DetectionTarget,
    ) -> torch.Tensor:

        positive_mask = target.objectness.bool()

        if not positive_mask.any():
            return prediction.boxes.sum() * 0

        prediction_boxes = prediction.boxes.permute(
            0,
            2,
            3,
            1,
        )

        prediction_boxes = prediction_boxes[
            positive_mask
        ]

        target_boxes = target.boxes[
            positive_mask
        ]

        assert prediction_boxes.ndim == 2
        assert prediction_boxes.shape[-1] == 4

        return self.box_loss_fn(
            prediction_boxes,
            target_boxes,
        )

    def _objectness_loss(
        self,
        prediction: DetectionPrediction,
        target: DetectionTarget,
    ) -> torch.Tensor:

        return self.objectness_loss_fn(
            prediction.objectness,
            target.objectness.float(),
        )

    def _classification_loss(
        self,
        prediction: DetectionPrediction,
        target: DetectionTarget,
    ) -> torch.Tensor:

        positive_mask = target.objectness.bool()

        if not positive_mask.any():
            return prediction.classes.sum() * 0

        prediction_classes = prediction.classes.permute(
            0,
            2,
            3,
            1,
        )

        prediction_classes = prediction_classes[
            positive_mask
        ]

        target_classes = target.classes[
            positive_mask
        ]

        assert prediction_classes.ndim == 2

        return self.classification_loss_fn(
            prediction_classes,
            target_classes,
        )