from torch import nn
from torch.optim import Optimizer

from trainer.train_one_epoch import train_one_epoch
from trainer.validate import validate


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        validation_loader,
        optimizer: Optimizer,
        criterion: nn.Module,
        encoder,
        device,
        epochs: int,
        logger=None,
        checkpoint_manager=None,
        scheduler=None,
    ):
        self.model = model

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.optimizer = optimizer
        self.criterion = criterion
        self.encoder = encoder

        self.device = device
        self.epochs = epochs

        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler

        self.best_loss = float("inf")

    def fit(self):

        self.model.to(self.device)

        for epoch in range(1, self.epochs + 1):

            train_metrics = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                encoder=self.encoder,
                device=self.device,
            )

            validation_metrics = validate(
                model=self.model,
                dataloader=self.validation_loader,
                criterion=self.criterion,
                encoder=self.encoder,
                device=self.device,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self._print_metrics(
                epoch,
                train_metrics,
                validation_metrics,
            )

            if self.logger is not None:

                self.logger.log(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                )

            if self.checkpoint_manager is not None:

                self.checkpoint_manager.save_last(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                )

                if validation_metrics["loss"] < self.best_loss:

                    self.best_loss = validation_metrics["loss"]

                    self.checkpoint_manager.save_best(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                    )

    @staticmethod
    def _print_metrics(
        epoch,
        train_metrics,
        validation_metrics,
    ):

        print(
            f"Epoch [{epoch}]"
        )

        print(
            f"Train Loss: {train_metrics['loss']:.4f}"
        )

        print(
            f"Validation Loss: {validation_metrics['loss']:.4f}"
        )

        print("-" * 60)