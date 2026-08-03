from pathlib import Path
import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = 28
    NUM_CLASSES = 10

    BATCH_SIZE = 64
    NUM_WORKERS = 4

    DATA_DIR = Path("data")

    DEVICE = "cuda"