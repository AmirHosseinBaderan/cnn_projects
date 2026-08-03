import torch

from diffusion.beta_scheduler import linear_beta_schedule


class NoiseScheduler:

    def __init__(
        self,
        num_timesteps: int = 1000,
        device="cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = linear_beta_schedule(
            num_timesteps
        ).to(device)

        self.alphas = 1 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0,
        )


    def add_noise(
        self,
        images,
        timesteps,
    ):
        noise = torch.randn_like(images)

        alpha_bar = self.alpha_bars[timesteps]

        alpha_bar = alpha_bar.view(
            -1,
            1,
            1,
            1,
        )

        noisy_images = (
            torch.sqrt(alpha_bar) * images
            +
            torch.sqrt(1 - alpha_bar) * noise
        )

        return noisy_images, noise
    
    @torch.no_grad()
    def remove_noise(
        self,
        x,
        predicted_noise,
        timestep,
    ):
    
        beta = self.betas[timestep].view(-1, 1, 1, 1)
        alpha = self.alphas[timestep].view(-1, 1, 1, 1)
        alpha_bar = self.alpha_bars[timestep].view(-1, 1, 1, 1)
    
        mean = (
            1.0 / torch.sqrt(alpha)
        ) * (
            x
            - (beta / torch.sqrt(1 - alpha_bar))
            * predicted_noise
        )
    
        if timestep.item() > 0:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(beta) * noise
    
        return mean