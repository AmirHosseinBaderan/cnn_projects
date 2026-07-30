import torch
from pathlib import Path

import cv2
import numpy as np

from config import Config
from dataset.vocabulary import Vocabulary
from models.recognizer import CRNN
from trainer.checkpoint import CheckpointManager

from inference.preprocess import ImagePreprocessor
from inference.predictor import Predictor
from inference.line_detector import LineDetector

from preprocessing.transforms import valid_transform

from decoder.greedy import GreedyDecoder


class OCREngine:

    def __init__(
        self,
        checkpoint_path: str,
        vocabulary_file: str,
        device: str | None = None,
    ):

        self.device = device or Config.DEVICE

        # Vocabulary
        self.vocabulary = Vocabulary(
            vocab_file=vocabulary_file
        )

        # Model
        self.model = CRNN(
            num_classes=self.vocabulary.num_classes
        )

        # Load Checkpoint
        CheckpointManager.load(
            model=self.model,
            optimizer=None,
            path=checkpoint_path,
            device=self.device,
        )

        self.model.to(self.device)
        self.model.eval()

        # Decoder
        self.decoder = GreedyDecoder(
            vocabulary=self.vocabulary
        )

        # Line Detector
        self.line_detector = LineDetector()

        # Image Preprocessor
        self.preprocessor = ImagePreprocessor(
            transform=valid_transform
        )

        # Predictor
        self.predictor = Predictor(
            model=self.model,
            decoder=self.decoder,
            device=self.device,
        )

    def predict(
        self,
        image,
    ) -> str:
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise FileNotFoundError(
                    f"Image not found: {image}"
                )

        # Detect text lines
        lines = self.line_detector.detect(image)

        # No line found
        if len(lines) == 0:
            return ""

        texts = []

        # OCR each line
        for line in lines:
            tensor = self.preprocessor.preprocess(
                line["image"]
            )
            text = self.predictor.predict(
                tensor
            )
            texts.append(text)

        # Merge lines
        return "\n".join(texts)