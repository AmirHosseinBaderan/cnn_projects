import torch
import os
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
from utils.logger import logger


def train():
    device = torch.device(Config.DEVICE)

    if device.type == "cpu":
        compute_threads = max(1, os.cpu_count() - Config.NUM_WORKERS)
        torch.set_num_threads(compute_threads)
        torch.set_num_interop_threads(1)

        logger.info(
            f"CPU device detected : "
            f"using {compute_threads} compute threads, "
            f"{torch.get_num_interop_threads()} interop threads"
        )

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
        persistent_workers=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=CTCCollate(),
        persistent_workers=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
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

    tensor_logger = TensorBoardLogger(
        log_dir=Config.LOG_DIR
    )

    checkpoint = CheckpointManager(
        checkpoint_dir=Config.CHECKPOINT_DIR
    )

    # Load checkpoint if exists
    start_epoch = 0
    last_checkpoint_path = checkpoint.checkpoint_dir / "last.pt"
    best_checkpoint_path = checkpoint.checkpoint_dir / "best.pt"

    if last_checkpoint_path.exists():
        start_epoch = CheckpointManager.load(
            model,
            last_checkpoint_path,
            device,
            optimizer
        )
    elif best_checkpoint_path.exists():
        start_epoch = CheckpointManager.load(
            model,
            best_checkpoint_path,
            device,
            optimizer
        )

    logger.info(
        f"Resuming from epoch {start_epoch}"
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        metrics=metrics,
        logger=tensor_logger,
        checkpoint=checkpoint,
    )

    trainer.fit(
        epochs=Config.EPOCHS,
        start_epoch=start_epoch
    )


if __name__ == "__main__":
    train()