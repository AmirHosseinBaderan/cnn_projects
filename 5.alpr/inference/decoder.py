import torch

from domain import BoundingBox


class DetectionDecoder:

    def __init__(
        self,
        image_size: int = 640,
        stride: int = 32,
        score_threshold: float = 0.1,
    ):
        self.image_size = image_size
        self.stride = stride
        self.score_threshold = score_threshold

    def decode(
        self,
        prediction,
    ):

        detections = []

        boxes = prediction.boxes[0]
        objectness = prediction.objectness[0].sigmoid()

        grid_size = objectness.shape[0]

        for gy in range(grid_size):

            for gx in range(grid_size):

                score = objectness[
                    gy,
                    gx,
                ].item()

                if score < self.score_threshold:
                    continue

                tx = boxes[
                    0,
                    gy,
                    gx,
                ].item()

                ty = boxes[
                    1,
                    gy,
                    gx,
                ].item()

                tw = boxes[
                    2,
                    gy,
                    gx,
                ].item()

                th = boxes[
                    3,
                    gy,
                    gx,
                ].item()

                cx = (
                    gx + tx
                ) * self.stride

                cy = (
                    gy + ty
                ) * self.stride

                width = tw * self.image_size
                height = th * self.image_size

                xmin = cx - width / 2
                ymin = cy - height / 2
                xmax = cx + width / 2
                ymax = cy + height / 2

                detections.append({
                    "score": score,
                    "bbox": BoundingBox(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    ),
                })

        print(f"[DEBUG] Decoder found {len(detections)} detections")
        return detections