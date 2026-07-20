from torchvision import transforms
from preprocessing.resize_height import ResizeHeight
from config import Config
from preprocessing.augmentations import (
    RandomRotation,
    RandomBrightnessContrast,
    RandomGaussianNoise,
    RandomGaussianBlur,
    RandomMotionBlur,
    RandomPerspective,
)

train_transform = transforms.Compose([
    ResizeHeight(Config.IMAGE_HEIGHT),

    RandomRotation(),
    RandomBrightnessContrast(),
    RandomGaussianNoise(),
    RandomGaussianBlur(),
    RandomMotionBlur(),
    RandomPerspective(),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])

valid_transform = transforms.Compose([
    ResizeHeight(Config.IMAGE_HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])
