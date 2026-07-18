import cv2
import time

from src.config import Config
from src.model import FaceEmbeddingNet
from src.detector import FaceDetector
from src.database import FaceDatabase
from src.search import FaceSearch
from src.recognizer import FaceRecognizer
from src.visualizer import FaceVisualizer
from src.checkpoint import load_checkpoint


def main():
    database = FaceDatabase()
    data = database.load()

    search = FaceSearch(
        data
    )

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

    recognizer = FaceRecognizer(
        detector=detector,
        model=model,
        search_engine=search,
        device=Config.DEVICE
    )

    visualizer = FaceVisualizer()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Cannot open camera")
        return

    previous_time = time.time()
    fps = 0.0

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        results = recognizer.recognize(frame)

        frame = visualizer.draw_frame(
            frame,
            results
        )

        current_time = time.time()

        fps = 1.0 / (current_time - previous_time)

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Face Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()