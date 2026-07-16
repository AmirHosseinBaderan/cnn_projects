from src.detector import FaceDetector
from src.config import Config

detector = FaceDetector(
    Config.DEVICE
)

faces = detector.detect(
    "./data/test/group.jpg"
)

print(
    "Faces found:",
    len(faces)
)

for i, face in enumerate(faces):
    face.save(
        f"face_{i}.jpg"
    )
