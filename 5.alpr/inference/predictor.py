import cv2
import torch

from inference.preprocess import ImagePreprocessor
from config import Config


class Predictor:

    def __init__(
        self,
        model,
        checkpoint,
        device,
        image_size,
    ):
        print("[DEBUG] About to load checkpoint")
        self.model = model.to(device)
        self.device = device

        checkpoint.load(
            path=Config.BEST_MODEL_NAME,
            model=model,
            device=device
        )
        print("[DEBUG] Checkpoint loaded")

        self.model.eval()

        self.preprocessor = ImagePreprocessor(
            image_size=image_size,
        )
        print(f"[DEBUG] Predictor initialized. Preprocessor created.")

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
    ):
        print(f"[DEBUG] Predictor.predict called with image_path: {image_path}")

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = self.preprocessor.preprocess(
            image,
        )

        image = image.to(self.device)

        prediction = self.model(
            image,
        )

        return prediction