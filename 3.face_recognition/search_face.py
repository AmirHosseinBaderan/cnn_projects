import torch

from src.model import FaceEmbeddingNet
from src.config import Config
from src.checkpoint import load_checkpoint

from src.database import FaceDatabase
from src.search import FaceSearch
from src.evaluator import get_embedding
from src.transforms import FaceTransform

import cv2



def load_image(path):

    image = cv2.imread(path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    transform = FaceTransform()

    return transform(image)



model = FaceEmbeddingNet().to(
    Config.DEVICE
)


load_checkpoint(
    Config.CHECKPOINT_PATH,
    model
)



db = FaceDatabase()
data = db.load()
search_engine = FaceSearch(
    data
)


image = load_image(
    "./data/test/unknown.jpg"
)

query_embedding = get_embedding(
    model,
    image,
    Config.DEVICE
)
result = search_engine.search(
    query_embedding
)

print(result)