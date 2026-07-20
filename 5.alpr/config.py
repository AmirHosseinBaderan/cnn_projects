from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RESOURCE_DIR = PROJECT_ROOT / "resources"
CHECKPOINT_DIR = RESOURCE_DIR / "checkpoints"
LOG_DIR = RESOURCE_DIR / "logs"
TENSORBOARD_DIR = RESOURCE_DIR / "tensorboard"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
NUM_WORKERS = 4
PIN_MEMORY = True

# Image
IMAGE_SIZE = (224, 224)

# Random
SEED = 42