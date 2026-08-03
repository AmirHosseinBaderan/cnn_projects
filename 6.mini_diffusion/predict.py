import matplotlib.pyplot as plt
import torch

from inference.diffusion_predictor import DiffusionPredictor


predictor = DiffusionPredictor(
    checkpoint_path="checkpoints/last_checkpoint.pt"
)

while True:
    label = input("Give label : ")
    if label.lower() == "exit":
        print("---- exit -----")
        break
    
    try:
        int_label = int(label)
    except:
        print("just enter int or exit")
        continue
    
    image = predictor.generate(
        label=int_label
    )

    image = image.squeeze().cpu()


    plt.figure(figsize=(3,3))
    plt.imshow(
        image,
        cmap="gray"
    )
    plt.title(
        f"Generated Digit: {int_label}"
    )
    plt.axis("off")
    plt.show()