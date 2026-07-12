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