import torch
from torch.utils.data import DataLoader

from .model import FaceEmbeddingNet
from .triplet_dataset import TripletDataset
from .loss import TripletLoss
from .logger import logger
from .config import Config


def train():
    dataset = TripletDataset(
        "./data/raw"
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    model = FaceEmbeddingNet().to(Config.DEVICE)
    criterion = TripletLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        for anchor, positive, negative in loader:
            anchor = anchor.to(Config.DEVICE)
            positive = positive.to(Config.DEVICE)
            negative = negative.to(Config.DEVICE)

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = criterion(
                anchor_embedding,
                positive_embedding,
                negative_embedding,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(
            f"Epoch : {epoch+1}/{Config.EPOCHS} Loss : {total_loss/len(loader):.4f}"
        )

if __name__ == "__main__":
    train()