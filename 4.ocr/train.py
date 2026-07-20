import torch
from torch.utils.data import DataLoader

from config import Config

from dataset.synth90k import Synth90kDataset
from dataset.vocabulary import Vocabulary
from dataset.collate import CTCCollate

from preprocessing.transforms import (
    train_transform,
    valid_transform
)

from models.recognizer import CRNN

from trainer.loss import CTCLossWrapper
from trainer.metrics import OCRMetrics
from trainer.logger import TensorBoardLogger
from trainer.checkpoint import CheckpointManager
from trainer.trainer import Trainer


def train():
    device = torch.device(Config.DEVICE)

    vocab = Vocabulary(
        vocab_file="resources/vocab.json"
    )
    train_dataset = Synth90kDataset(
        image_dir="data/images",
        annotation_file="data/train.txt",
        vocabulary=vocab,
        transform=train_transform,
    )

    valid_dataset = Synth90kDataset(
        image_dir="data/images",
        annotation_file="data/valid.txt",
        vocabulary=vocab,
        transform=valid_transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=CTCCollate(),
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=CTCCollate(),
    )
    model = CRNN(
        num_classes=vocab.num_classes
    )

    criterion = CTCLossWrapper()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )

    metrics = OCRMetrics(
        vocabulary=vocab
    )

    logger = TensorBoardLogger(
        log_dir=Config.LOG_DIR
    )

    checkpoint = CheckpointManager(
        checkpoint_dir=Config.CHECKPOINT_DIR
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        metrics=metrics,
        logger=logger,
        checkpoint=checkpoint,
    )

    trainer.fit(
        epochs=Config.EPOCHS
    )


if __name__ == "__main__":
    train()