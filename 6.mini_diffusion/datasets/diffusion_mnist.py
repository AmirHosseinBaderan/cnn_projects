import torch

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from diffusion.noise_scheduler import NoiseScheduler


class DiffusionMNISTDataset(Dataset):

    def __init__(
        self,
        root,
        scheduler,
        train=True,
    ):

        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=ToTensor(),
        )

        self.scheduler = scheduler


    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):

        image, label = self.dataset[index]


        timestep = torch.randint(
            0,
            self.scheduler.num_timesteps,
            (1,)
        ).item()


        timestep_tensor = torch.tensor(
            [timestep]
        )


        noisy_image, noise = self.scheduler.add_noise(
            image.unsqueeze(0),
            timestep_tensor
        )


        noisy_image = noisy_image.squeeze(0)


        return {
            "image": noisy_image,
            "noise": noise.squeeze(0),
            "label": torch.tensor(label),
            "timestep": torch.tensor(timestep),
        }