import os

import torch


class Config:

    # Device
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Dataset
    IMAGE_SIZE = 640

    TRAIN_DIR = "data/car_images/train"
    TRAIN_IMAGE_DIR = "data/car_images/train/images"
    TRAIN_ANNOTATION_DIR = "data/car_images/train/annotations"

    VALIDATION_DIR = "data/car_images/validation"
    VALIDATION_IMAGE_DIR = "data/car_images/validation/images"
    VALIDATION_ANNOTATION_DIR = "data/car_images/validation/annotations"

    TEST_IMAGE_DIR = "data/car_images/test/images"
    TEST_ANNOTATION_DIR = "data/car_images/test/annotations"

    # DataLoader
    BATCH_SIZE = 8
    NUM_WORKERS = os.cpu_count() if DEVICE.type == "cpu" else 4
    PIN_MEMORY = torch.cuda.is_available()

    # Model
    NUM_CLASSES = 1
    STRIDE = 32

    # Training
    EPOCHS = 50

    LEARNING_RATE = 1e-3

    WEIGHT_DECAY = 1e-4

    # Loss
    BOX_LOSS_WEIGHT = 5.0
    OBJECTNESS_LOSS_WEIGHT = 1.0
    CLASSIFICATION_LOSS_WEIGHT = 1.0

    # Checkpoints
    CHECKPOINT_DIR = "./resources/checkpoints"

    BEST_MODEL_NAME = "best.pt"
    LAST_MODEL_NAME = "last.pt"

    # TensorBoard
    LOG_DIR = "./resources/tensorboard"