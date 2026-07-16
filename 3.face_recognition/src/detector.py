from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetector:

    def __init__(
            self,
            device
    ):

        self.detector = MTCNN(
            keep_all=True,
            device=device
        )

    def detect(
            self,
            image_path
    ):

        image = Image.open(
            image_path
        )

        boxes, probs = self.detector.detect(
            image
        )

        faces = []

        if boxes is None:
            return faces

        for box, prob in zip(
                boxes,
                probs
        ):

            if prob < 0.90:
                continue

            x1, y1, x2, y2 = box.astype(int)

            face = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            faces.append(
                face
            )
        return faces
