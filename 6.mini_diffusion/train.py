import torch

from diffusion.noise_scheduler import NoiseScheduler


scheduler = NoiseScheduler()


images = torch.randn(
    4,
    1,
    28,
    28
)


timesteps = torch.tensor(
    [10, 100, 500, 900]
)


noisy_images, noise = scheduler.add_noise(
    images,
    timesteps
)


print(noisy_images.shape)
print(noise.shape)