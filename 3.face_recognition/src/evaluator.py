import torch
import numpy as np

from PIL import Image

import torchvision.transforms as transforms


transform = transforms.Compose(
    [
        transforms.Resize(
            (112, 112)
        ),
        transforms.ToTensor(),
    ]
)


def get_embedding(
        model,
        image,
        device
):

    model.eval()

    if isinstance(
            image,
            Image.Image
    ):

        image = transform(
            image
        )

    elif isinstance(
            image,
            np.ndarray
    ):

        image = Image.fromarray(
            image
        )

        image = transform(
            image
        )

    elif torch.is_tensor(
            image
    ):

        pass

    else:

        raise TypeError(
            "Unsupported image type."
        )

    image = image.unsqueeze(
        0
    )

    image = image.to(
        device
    )

    with torch.no_grad():

        embedding = model(
            image
        )

    return embedding.squeeze(
        0
    ).cpu()