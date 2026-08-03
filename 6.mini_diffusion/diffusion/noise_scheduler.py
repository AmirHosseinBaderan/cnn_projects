import torch


class NoiseScheduler:

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
        )