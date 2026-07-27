import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config

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

from trainer.loss import DetectionLoss
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

    print("=" * 60)
    print("Running Smoke Test...")
    print("=" * 60)

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

    print(f"Images        : {tuple(images.shape)}")
    print(f"Boxes         : {tuple(prediction.boxes.shape)}")
    print(f"Objectness    : {tuple(prediction.objectness.shape)}")
    print(f"Classes       : {tuple(prediction.classes.shape)}")

    print()

    print(f"Loss               : {losses['loss']:.4f}")
    print(f"Box Loss           : {losses['box_loss']:.4f}")
    print(f"Objectness Loss    : {losses['objectness_loss']:.4f}")
    print(f"Classification Loss: {losses['classification_loss']:.4f}")

    print("=" * 60)


def main():

    device = build_device()

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

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        encoder=encoder,
        device=device,
        epochs=Config.EPOCHS,
    )

    trainer.fit()


if __name__ == "__main__":
    main()