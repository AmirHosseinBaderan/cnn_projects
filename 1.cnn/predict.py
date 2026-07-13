import torch

from models.cnn import CNN
from config import Config

model = CNN()

model.load_state_dict(
    torch.load(
        Config.MODEL_PATH,
        weights_only=True
    )
)

model.eval()
