import matplotlib.pyplot as plt


def plot_loss(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["valid_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid()

    plt.show()

def plot_accuracy(history):

    plt.figure(figsize=(8,5))

    plt.plot(
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["valid_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid()

    plt.show()

def show_images(dataset, count=8):

    plt.figure(figsize=(12, 4))

    for i in range(count):

        image, label = dataset[i]

        image = image.permute(1, 2, 0)

        plt.subplot(2, 4, i + 1)

        plt.imshow(image)

        plt.title(str(label))

        plt.axis("off")

    plt.tight_layout()
    plt.show()