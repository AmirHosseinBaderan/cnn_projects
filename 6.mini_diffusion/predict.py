import matplotlib.pyplot as plt
import torch

from inference.diffusion_predictor import DiffusionPredictor


predictor = DiffusionPredictor(
    checkpoint_path="checkpoints/best.pt"
)


image = predictor.generate(
    label=7
)

image = image.squeeze().cpu()


plt.figure(figsize=(3,3))

plt.imshow(
    image,
    cmap="gray"
)

plt.title(
    "Generated Digit: 7"
)

plt.axis("off")

plt.show()