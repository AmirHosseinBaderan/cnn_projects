import torch
import os

class Config:
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_WORKERS = min(4, max(1, os.cpu_count() // 2))
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "runs"
    PIN_MEMORY = DEVICE == "cuda"
    PREFETCH_FACTOR = 4