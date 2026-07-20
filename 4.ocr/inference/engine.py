import torch

from config import Config
from dataset.vocabulary import Vocabulary
from models.recognizer import CRNN
from trainer.checkpoint import CheckpointManager
from inference.preprocess import ImagePreprocessor
from inference.predictor import Predictor
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

        # Load Weights
        CheckpointManager.load(
            model=self.model,
            path=checkpoint_path,
            device=self.device,
        )

        # Decoder
        self.decoder = GreedyDecoder(
            vocabulary=self.vocabulary
        )


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
        tensor = self.preprocessor.preprocess(image)
        text = self.predictor.predict(tensor)

        return text