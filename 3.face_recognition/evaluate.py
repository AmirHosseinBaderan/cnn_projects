import cv2
import torch

from src.model import FaceEmbeddingNet
from src.config import Config
from src.checkpoint import load_checkpoint

from src.evaluator import (
    get_embedding,
    similarity
)

from src.eval_dataset import EvaluationDataset
from src.transforms import FaceTransform
from src.logger import logger


def load_image(path):
    image = cv2.imread(
        str(path)
    )
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    transform = FaceTransform()
    return transform(image)


def evaluate():
    logger.info("Start Evaluate ")

    model = FaceEmbeddingNet().to(
        Config.DEVICE
    )
    load_checkpoint(
        Config.CHECKPOINT_PATH,
        model
    )
    dataset = EvaluationDataset(
        "./data/raw"
    )
    same_scores = []
    different_scores = []

    for i in range(1000):
        img1, img2, label = dataset.create_pair()
        img1 = load_image(img1)
        img2 = load_image(img2)

        emb1 = get_embedding(
            model,
            img1,
            Config.DEVICE
        )
        emb2 = get_embedding(
            model,
            img2,
            Config.DEVICE
        )
        score = similarity(
            emb1,
            emb2
        )
        if label == 1:
            same_scores.append(score)

        else:
            different_scores.append(score)

    logger.info("====================")
    logger.info(
        f"Same Average: {sum(same_scores) / len(same_scores)}"
    )
    logger.info(
        f"Different Average: {sum(different_scores) / len(different_scores)}"
    )
    logger.info("====================")


if __name__ == "__main__":
    evaluate()
