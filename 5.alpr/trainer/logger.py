from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:

    def __init__(self, log_dir):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))

    def log(self, epoch, train_metrics, validation_metrics):

        self.writer.add_scalar(
            "Loss/Train",
            train_metrics["loss"],
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Validation",
            validation_metrics["loss"],
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Box/Train",
            train_metrics["box_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Box/Validation",
            validation_metrics["box_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Objectness/Train",
            train_metrics["objectness_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Objectness/Validation",
            validation_metrics["objectness_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Classification/Train",
            train_metrics["classification_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Classification/Validation",
            validation_metrics["classification_loss"],
            epoch,
        )

    def log_learning_rate(self, lr, epoch):

        self.writer.add_scalar(
            "Learning Rate",
            lr,
            epoch,
        )

    def flush(self):

        self.writer.flush()

    def close(self):

        self.writer.close()
