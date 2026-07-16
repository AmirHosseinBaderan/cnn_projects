import torch

from src.model import FaceEmbeddingNet
from src.dataset import FaceDataset
from src.evaluator import get_embedding
from src.checkpoint import load_checkpoint
from src.triplet_dataset import TripletDataset

from src.config import Config

model = FaceEmbeddingNet().to(
    Config.DEVICE
)

load_checkpoint(
    Config.CHECKPOINT_PATH,
    model
)

dataset = TripletDataset(
    "./data/raw"
)


anchor, positive, negative = dataset[0]
anchor_embedding = get_embedding(
    model,
    anchor,
    Config.DEVICE
)


positive_embedding = get_embedding(
    model,
    positive,
    Config.DEVICE
)


negative_embedding = get_embedding(
    model,
    negative,
    Config.DEVICE
)
same = torch.nn.functional.cosine_similarity(
    anchor_embedding,
    positive_embedding
)


different = torch.nn.functional.cosine_similarity(
    anchor_embedding,
    negative_embedding
)


print(
    "Same person:",
    same.item()
)


print(
    "Different person:",
    different.item()
)