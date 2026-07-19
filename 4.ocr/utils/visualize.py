import matplotlib.pyplot as plt


def show_batch(batch, count=8):
    images = batch["images"]
    fig, axes = plt.subplots(2,4, figsize=(12,5))

    axes = axes.flatten()

    for i in range(min(count,len(images))):
        image = images[i].squeeze().cpu().numpy()

        image = image * 0.5 + 0.5
        axes[i].imshow(
            image,
            cmap="gray"
        )

        axes[i].axis("off")
        axes[i].set_title(
            batch["targets"][i],
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()