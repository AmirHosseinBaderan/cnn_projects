import argparse
import time
import cv2

from src.config import Config
from src.detector import FaceDetector
from src.database import FaceDatabase
from src.evaluator import get_embedding
from src.model import FaceEmbeddingNet
from src.checkpoint import load_checkpoint


CAPTURE_COUNT = 10


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        required=True
    )

    args = parser.parse_args()

    model = FaceEmbeddingNet().to(
        Config.DEVICE
    )

    load_checkpoint(
        Config.CHECKPOINT_PATH,
        model
    )

    model.eval()

    detector = FaceDetector(
        Config.DEVICE
    )

    database = FaceDatabase()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Cannot open camera")
        return

    captured = 0

    print()
    print("===================================")
    print(f"Register : {args.name}")
    print("Press SPACE to capture")
    print("Press Q to quit")
    print("===================================")

    while captured < CAPTURE_COUNT:

        ret, frame = camera.read()

        if not ret:
            break

        detections = detector.detect(
            frame
        )

        display = frame.copy()

        for detection in detections:

            x1, y1, x2, y2 = detection["box"]

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        cv2.putText(
            display,
            f"{args.name}   {captured}/{CAPTURE_COUNT}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Register Face",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        if key != ord(" "):

            continue

        if len(detections) != 1:

            print("Exactly one face must be visible.")

            continue

        embedding = get_embedding(
            model=model,
            image=detections[0]["face"],
            device=Config.DEVICE
        )

        database.add_face(
            args.name,
            embedding
        )

        captured += 1

        print(
            f"Captured {captured}/{CAPTURE_COUNT}"
        )

        time.sleep(0.5)

    camera.release()

    cv2.destroyAllWindows()

    print()

    print("Finished")


if __name__ == "__main__":
    main()