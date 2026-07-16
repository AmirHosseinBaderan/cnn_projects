import torch

from src.model import FaceEmbeddingNet
from src.config import Config
from src.checkpoint import load_checkpoint

from src.database import FaceDatabase
from src.register import FaceRegister

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
    "./data/register/Ali"
)
database = FaceDatabase()

database.add_face(
    "Ali",
    embedding
)
print(
    "Face Registered"
)
