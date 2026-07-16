import torch

class Config:
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    CHECKPOINT_PATH = (
        "checkpoints/"
        "face_embedding.pth"
    )