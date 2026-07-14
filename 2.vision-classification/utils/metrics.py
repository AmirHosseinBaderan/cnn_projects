from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


def get_confusion_matrix(
        labels,
        predictions
):
    return confusion_matrix(
        labels,
        predictions
    )


def plot_confusion_matrix(
        cm,
        class_names
):
    fig, ax = plt.subplots(figsize=(8, 8))

    image = ax.imshow(cm, cmap="Blues")

    # Color bar
    fig.colorbar(image)

    # Ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    # Labels
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Axis titles
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Figure title
    ax.set_title("Confusion Matrix")

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Write numbers inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()
    plt.show()


def print_classification_report(
        all_labels,
        all_predictions,
        class_names
):
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )

    print(report)