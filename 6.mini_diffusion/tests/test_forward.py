import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor

from diffusion.noise_scheduler import NoiseScheduler


dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


image, label = dataset[0]


image = image.unsqueeze(0)


scheduler = NoiseScheduler()


timesteps = [
    0,
    100,
    300,
    500,
    700,
    999,
]


fig, axes = plt.subplots(
    1,
    len(timesteps),
    figsize=(12,3)
)


for idx, t in enumerate(timesteps):

    timestep = torch.tensor([t])

    noisy_image, _ = scheduler.add_noise(
        image,
        timestep
    )

    axes[idx].imshow(
        noisy_image.squeeze(),
        cmap="gray"
    )

    axes[idx].set_title(
        f"t={t}"
    )

    axes[idx].axis("off")


plt.show()