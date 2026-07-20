import torch

from trainer.train_one_epoch import train_one_epoch
from trainer.validate import validate
from utils.logger import logger

class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        metrics,
        logger,
        checkpoint,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.metrics = metrics
        self.logger = logger
        self.checkpoint = checkpoint

    def fit(self, epochs):

        self.model.to(self.device)

        for epoch in range(epochs):

            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device
            )

            validation = validate(
                model=self.model,
                dataloader=self.valid_loader,
                criterion=self.criterion,
                device=self.device,
                metrics=self.metrics
            )

            valid_loss = validation["loss"]

            predictions = validation["predictions"]

            logger.info(
                f"\nEpoch {epoch + 1}/{epochs}"
            )

            logger.info(
                f"Train Loss : {train_loss:.4f}"
            )

            logger.info(
                f"Valid Loss : {valid_loss:.4f}"
            )

            self.logger.log_train_loss(
                train_loss,
                epoch + 1
            )

            self.logger.log_validation_loss(
                valid_loss,
                epoch + 1
            )

            self.logger.log_learning_rate(
                self.optimizer.param_groups[0]["lr"],
                epoch + 1
            )

            self.logger.log_metrics(
                predictions,
                epoch + 1
            )

            self.logger.log_model_weights(
                self.model,
                epoch + 1
            )

            self.checkpoint.save_last(
                self.model,
                self.optimizer,
                epoch + 1,
                valid_loss
            )

            self.checkpoint.save_best(
                self.model,
                self.optimizer,
                epoch + 1,
                valid_loss
            )

        self.logger.close()