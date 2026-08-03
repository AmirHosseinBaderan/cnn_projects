import torch
import torch.nn as nn


class Trainer:

    def __init__(
        self,
        model,
        scheduler,
        optimizer,
        device,
    ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device

        self.loss_fn = nn.MSELoss()


    def _step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        batch_size = images.shape[0]

        timesteps = torch.randint(
            0,
            self.scheduler.num_timesteps,
            (batch_size,),
            device=self.device,
        )

        noisy_images, noise = self.scheduler.add_noise(images, timesteps)

        predicted_noise = self.model(noisy_images, timesteps, labels)

        loss = self.loss_fn(predicted_noise, noise)

        return loss


    def train_step(
        self,
        batch,
    ):
        loss = self._step(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(
        self,
        batch,
    ):
        with torch.no_grad():
            loss = self._step(batch)
        return loss.item()