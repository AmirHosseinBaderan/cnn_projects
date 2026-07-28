from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config

from utils.logger import logger

from dataset.collate import DetectorCollate
from dataset.detector_dataset import DetectorDataset
from dataset.parsers import IRLPRXMLParser

from models.detector import Detector

from preprocessing import (
    Resize,
    ToTensor,
    Normalize,
)
from preprocessing.compose import Compose

from trainer.checkpoint import CheckpointManager
from trainer.loss import DetectionLoss
from trainer.logger import TensorBoardLogger
from trainer.target_encoder import TargetEncoder
from trainer.trainer import Trainer


def build_device():
    return Config.DEVICE


def build_parser():
    return IRLPRXMLParser()


def build_transforms():
    return Compose([
        Resize(
            width=Config.IMAGE_SIZE,
            height=Config.IMAGE_SIZE,
        ),
        ToTensor(),
        Normalize(),
    ])


def build_dataset(root):
    return DetectorDataset(
        root=root,
        parser=build_parser(),
        transform=build_transforms(),
    )


def build_dataloader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS,
        collate_fn=DetectorCollate(),
    )


def build_model():

    return Detector()


def build_optimizer(model):

    return Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )


def build_loss():

    return DetectionLoss(
        box_weight=Config.BOX_LOSS_WEIGHT,
        objectness_weight=Config.OBJECTNESS_LOSS_WEIGHT,
        classification_weight=Config.CLASSIFICATION_LOSS_WEIGHT,
    )


def build_encoder():

    return TargetEncoder(
        image_size=Config.IMAGE_SIZE,
        stride=Config.STRIDE,
        num_classes=Config.NUM_CLASSES,
    )


def smoke_test(
    model,
    dataloader,
    encoder,
    criterion,
    device,
):
    logger.info("=" * 60)
    logger.info("Running Smoke Test...")
    logger.info("=" * 60)

    model.to(device)
    model.eval()

    batch = next(iter(dataloader))

    images = batch["images"]
    annotations = batch["annotations"]

    target = encoder.encode(
        annotations,
    ).to(device)

    with torch.no_grad():

        prediction = model(images)

        losses = criterion(
            prediction,
            target,
        )

    logger.info(f"Images        : {tuple(images.shape)}")
    logger.info(f"Boxes         : {tuple(prediction.boxes.shape)}")
    logger.info(f"Objectness    : {tuple(prediction.objectness.shape)}")
    logger.info(f"Classes       : {tuple(prediction.classes.shape)}")

    logger.info("")

    logger.info(f"Loss               : {losses['loss']:.4f}")
    logger.info(f"Box Loss           : {losses['box_loss']:.4f}")
    logger.info(f"Objectness Loss    : {losses['objectness_loss']:.4f}")
    logger.info(f"Classification Loss: {losses['classification_loss']:.4f}")

    logger.info("=" * 60)


def main():

    device = build_device()

    if device.type == "cpu":
        torch.set_num_threads(Config.TORCH_NUM_THREADS)
        torch.set_num_interop_threads(Config.TORCH_NUM_INTEROP_THREADS)
        logger.info(f"Torch threads: {torch.get_num_threads()}")
        logger.info(f"Torch interop threads: {torch.get_num_interop_threads()}")

    train_dataset = build_dataset(
        Config.TRAIN_DIR,
    )

    validation_dataset = build_dataset(
        Config.VALIDATION_DIR,
    )

    train_loader = build_dataloader(
        train_dataset,
        shuffle=True,
    )

    validation_loader = build_dataloader(
        validation_dataset,
        shuffle=False,
    )

    logger.info(f"Num Workers: {Config.NUM_WORKERS}")

    model = build_model()

    optimizer = build_optimizer(
        model,
    )

    criterion = build_loss()

    encoder = build_encoder()

    smoke_test(
        model=model,
        dataloader=train_loader,
        encoder=encoder,
        criterion=criterion,
        device=device,
    )

    tensorboard_logger = TensorBoardLogger(
        log_dir=Config.LOG_DIR,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Config.CHECKPOINT_DIR,
    )

    start_epoch = 0

    last_checkpoint = Path(
        Config.CHECKPOINT_DIR,
        Config.LAST_MODEL_NAME,
    )

    best_checkpoint = Path(
        Config.CHECKPOINT_DIR,
        Config.BEST_MODEL_NAME,
    )

    if last_checkpoint.exists():
        start_epoch = checkpoint_manager.load(
            model,
            Config.LAST_MODEL_NAME,
            device,
            optimizer,
        )
    elif best_checkpoint.exists():
        start_epoch = checkpoint_manager.load(
            model,
            Config.BEST_MODEL_NAME,
            device,
            optimizer,
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        encoder=encoder,
        device=device,
        epochs=Config.EPOCHS,
        logger=tensorboard_logger,
        checkpoint_manager=checkpoint_manager,
        scheduler=None,
        patience=Config.PATIENCE,
    )

    trainer.fit(start_epoch=start_epoch)

    tensorboard_logger.close()


if __name__ == "__main__":
    main()