import torch

from .model import FaceEmbeddingNet
from .config import Config
from .checkpoint import load_checkpoint

from .database import FaceDatabase
from .register import FaceRegister

model = FaceEmbeddingNet().to(
    Config.DEVICE
)
load_checkpoint(
    Config.CHECKPOINT_PATH,
    model
)
register = FaceRegister(
    model,
    Config.DEVICE
)
embedding = register.create_embedding(
    "./data/register/Amir"
)
database = FaceDatabase()

database.save(
    embedding,
    {
        "0":"Amir"
    }
)
print(
    "Face Registered"
)