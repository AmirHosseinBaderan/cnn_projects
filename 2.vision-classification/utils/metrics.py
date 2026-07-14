from sklearn.metrics import confusion_matrix

def get_confusion_matrix(
    labels,
    predictions
):
    return confusion_matrix(
        labels,
        predictions
    )