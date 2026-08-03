import torch

from models.unet import UNet
from diffusion.noise_scheduler import NoiseScheduler
from configs.config import Config
from utils.checkpoint_manager import CheckpointManager


class DiffusionPredictor:

    def __init__(
        self,
        checkpoint_path: str,
    ):

        self.device = Config.DEVICE


        self.model = UNet().to(
            self.device
        )


        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Config.CHECKPOINT_DIR,
            model=self.model,
            optimizer=None,
            scheduler=None,
        )


        self.checkpoint_manager.load_checkpoint(
            checkpoint_path
        )


        self.model.eval()


        self.scheduler = NoiseScheduler()



    @torch.no_grad()
    def generate(
        self,
        label: int,
        steps: int = 1000,
    ):

        label = torch.tensor(
            [label],
            device=self.device,
            dtype=torch.long,
        )


        image = torch.randn(
            1,
            1,
            28,
            28,
            device=self.device,
        )


        image = self.reverse_diffusion(
            image,
            label,
            steps,
        )


        return image



    def reverse_diffusion(
        self,
        image,
        label,
        steps,
    ):


        for timestep in reversed(
            range(steps)
        ):

            t = torch.tensor(
                [timestep],
                device=self.device,
                dtype=torch.long,
            )


            predicted_noise = self.model(
                image,
                t,
                label,
            )


            image = self.scheduler.remove_noise(
                image,
                predicted_noise,
                t,
            )


        return image