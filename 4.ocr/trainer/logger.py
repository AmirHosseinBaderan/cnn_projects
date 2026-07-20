from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:

    def __init__(self, log_dir):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))

    def log_train_loss(self, loss, epoch):
        self.writer.add_scalar(
            "Loss/Train",
            loss,
            epoch
        )

    def log_validation_loss(self, loss, epoch):
        self.writer.add_scalar(
            "Loss/Validation",
            loss,
            epoch
        )

    def log_learning_rate(self, lr, epoch):
        self.writer.add_scalar(
            "Learning Rate",
            lr,
            epoch
        )

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(
            name,
            value,
            step
        )

    def log_images(
            self,
            tag,
            images,
            epoch
    ):
        self.writer.add_images(
            tag,
            images,
            epoch
        )

    def log_predictions(
            self,
            gt_texts,
            pred_texts,
            epoch,
            max_samples=5
    ):

        lines = []

        count = min(
            len(gt_texts),
            max_samples
        )

        for i in range(count):

            lines.append(
                f"{i+1}. GT : {gt_texts[i]}"
            )

            lines.append(
                f"   PR : {pred_texts[i]}"
            )

            lines.append("")

        self.writer.add_text(
            "Predictions",
            "\n".join(lines),
            epoch
        )

    def log_model_weights(
            self,
            model,
            epoch
    ):

        for name, param in model.named_parameters():

            self.writer.add_histogram(
                name,
                param,
                epoch
            )

    def log_graph(
            self,
            model,
            sample_input
    ):

        try:
            self.writer.add_graph(
                model,
                sample_input
            )
        except Exception:
            pass

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
        
    def log_metrics(
        self,
        results,
        epoch
    ):

        text = ""
    
        for i, item in enumerate(results):
        
            text += f"{i+1}. GT : {item['gt']}\n"
            text += f"   PR : {item['pred']}\n\n"
    
        self.writer.add_text(
            "OCR Predictions",
            text,
            epoch
        )