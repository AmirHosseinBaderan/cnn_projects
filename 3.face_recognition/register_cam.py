import argparse
import time
import cv2

from src.config import Config
from src.detector import FaceDetector
from src.database import FaceDatabase
from src.evaluator import get_embedding
from src.model import FaceEmbeddingNet
from src.checkpoint import load_checkpoint
from src.logger import logger


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
        logger.error("Cannot open camera")
        return

    captured = 0

    logger.info(f"Register : {args.name}")
    logger.info("Press SPACE to capture")
    logger.info("Press Q to quit")
    
    cv2.namedWindow("Register Face", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Register Face", 1280, 720)

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
            logger.warning("Exactly one face must be visible.")
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
        logger.info(
            f"Captured {captured}/{CAPTURE_COUNT}"
        )
        time.sleep(0.5)

    camera.release()
    cv2.destroyAllWindows()

    logger.info("Finished")


if __name__ == "__main__":
    main()