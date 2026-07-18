import cv2
import numpy as np

from PIL import Image

import torchvision.transforms as transforms

from facenet_pytorch import MTCNN


class FaceDetector:

    def __init__(
            self,
            device,
            image_size=112,
            threshold=0.90
    ):

        self.threshold = threshold

        self.detector = MTCNN(
            image_size=image_size,
            keep_all=True,
            device=device
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size)
                ),
                transforms.ToTensor(),
            ]
        )

    def detect(
            self,
            image
    ):

        image = self._to_pil(
            image
        )

        boxes, probabilities = self.detector.detect(
            image
        )

        detections = []

        if boxes is None:
            return detections

        for box, probability in zip(
                boxes,
                probabilities
        ):

            if probability < self.threshold:
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            face = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            face = self.transform(
                face
            )

            detections.append(
                {
                    "face": face,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "confidence": float(probability),
                    "center": (
                        (x1 + x2) // 2,
                        (y1 + y2) // 2
                    ),
                    "size": (
                        x2 - x1,
                        y2 - y1
                    )
                }
            )

        return detections

    def _to_pil(
            self,
            image
    ):

        if isinstance(
                image,
                str
        ):

            return Image.open(
                image
            ).convert(
                "RGB"
            )

        if isinstance(
                image,
                np.ndarray
        ):

            return Image.fromarray(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )
            )

        if isinstance(
                image,
                Image.Image
        ):

            return image.convert(
                "RGB"
            )

        raise TypeError(
            "Unsupported image type."
        )