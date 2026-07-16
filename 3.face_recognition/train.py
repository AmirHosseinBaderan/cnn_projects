import torch

from torch.utils.data import DataLoader

from src.model import FaceEmbeddingNet
from src.triplet_dataset import TripletDataset
from src.loss import TripletLoss

from src.logger import logger
from src.config import Config

from src.checkpoint import (
    save_checkpoint,
    load_checkpoint
)



def train():

    logger.info(
        "Start training loop"
    )

    logger.info(
        f"Running on {Config.DEVICE}"
    )


    dataset = TripletDataset(
        "./data/raw"
    )


    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )


    model = FaceEmbeddingNet().to(
        Config.DEVICE
    )


    criterion = TripletLoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE
    )


    start_epoch = load_checkpoint(
        Config.CHECKPOINT_PATH,
        model,
        optimizer
    )
    best_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(
        start_epoch,
        Config.EPOCHS
    ):
        logger.info(
            f"Start epoch : {epoch + 1}"
        )
        model.train()
        total_loss = 0

        for anchor, positive, negative in loader:
            anchor = anchor.to(
                Config.DEVICE
            )
            positive = positive.to(
                Config.DEVICE
            )
            negative = negative.to(
                Config.DEVICE
            )
            anchor_embedding = model(anchor)

            positive_embedding = model(positive)

            negative_embedding = model(negative)

            loss = criterion(
                anchor_embedding,
                positive_embedding,
                negative_embedding
            )
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(loader)
        logger.info(
            f"Epoch : {epoch + 1}/{Config.EPOCHS} "
            f"Loss : {epoch_loss:.4f}"
        )
        # Save best model
        if epoch_loss < best_loss:
            early_stopping_counter = 0
            best_loss = epoch_loss
            save_checkpoint(
                model,
                Config.CHECKPOINT_PATH,
                epoch,
                optimizer,
                epoch_loss
            )

            logger.info(
                "Best model saved"
            )
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= Config.PATIENCE:
            logger.info("Early stopping")
            logger.info(
                f"Validation loss did not improve for {Config.PATIENCE} epochs. Stopping training."
            )
            break


if __name__ == "__main__":
    train()