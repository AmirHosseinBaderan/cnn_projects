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
            image_path
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

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
                    "confidence": float(probability)
                }
            )

        return detections