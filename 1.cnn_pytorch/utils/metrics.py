import torch
from sklearn.metrics import confusion_matrix

def accuracy(outputs, labels):

    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()

    total = labels.size(0)

    return correct / total


def get_confusion_matrix(
    labels,
    predictions
):
    return confusion_matrix(
        labels,
        predictions
    )