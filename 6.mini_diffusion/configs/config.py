from pathlib import Path
import torch

class Config:
    DEVICE = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    IMAGE_SIZE = 28
    NUM_CLASSES = 10

    BATCH_SIZE = 64
    NUM_WORKERS = 4

    EPOCHS = 10
    PATIENCE = 3

    DATA_DIR = Path("data")