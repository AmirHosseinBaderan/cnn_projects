import torch
from utils.metrics import get_confusion_matrix

def validate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_labels.extend(
                labels.cpu().numpy()
            )
            all_predictions.extend(
                predicted.cpu().numpy()
            )

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    cm = get_confusion_matrix(
        all_labels,
        all_predictions
    )

    return epoch_loss, accuracy,cm