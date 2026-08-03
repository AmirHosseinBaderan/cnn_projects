import torch

from models.unet import UNet
from diffusion.noise_scheduler import NoiseScheduler


class DiffusionPredictor:

    def __init__(
        self,
        checkpoint_manager,
        checkpoint_name,
        device=None,
    ):

        self.device = device

        self.model = UNet().to(
            self.device
        )


        checkpoint_manager.load(
            self.model,
            checkpoint_name
        )


        self.model.eval()


        self.scheduler = NoiseScheduler(
            device=self.device
        )


    def generate(
        self,
        label: int,
        batch_size: int = 1,
    ):

        labels = torch.full(
            (
                batch_size,
            ),
            label,
            device=self.device,
            dtype=torch.long,
        )


        noise = torch.randn(
            batch_size,
            1,
            28,
            28,
            device=self.device,
        )


        image = self.sample(
            noise,
            labels,
        )


        return image



    @torch.no_grad()
    def sample(
        self,
        noise,
        labels,
    ):

        image = noise


        # TODO:
        # reverse diffusion loop


        timestep = torch.tensor(
            [
                self.scheduler.num_timesteps - 1
            ],
            device=self.device,
        )


        predicted_noise = self.model(
            image,
            timestep,
            labels,
        )


        return predicted_noise