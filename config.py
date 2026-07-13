import torch


class Config:

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    BATCH_SIZE = 64

    NUM_EPOCHS = 20

    LEARNING_RATE = 0.001

    NUM_CLASSES = 10

    IMAGE_SIZE = 32

    NUM_WORKERS = 4

    PIN_MEMORY = True

    MODEL_PATH = "checkpoints/best_model.pth"
    MODEL_NAME = "cnn"