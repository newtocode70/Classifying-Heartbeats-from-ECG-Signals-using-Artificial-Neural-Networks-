import torch

def accuracy(y_true, y_pred):
    """Return overall classification accuracy as a fraction."""
    return (y_true == y_pred).float().mean().item()


def make_confusion_matrix(y_true, y_pred, number_of_classes=5):
    """Rows are true classes; columns are predicted classes."""
    matrix = torch.zeros(
        (number_of_classes, number_of_classes), dtype=torch.int64
    )

    for i in range(len(y_true)):
        actual = y_true[i].item()
        predicted = y_pred[i].item()
        matrix[actual, predicted] += 1

    return matrix


def per_class_accuracy(confusion_matrix):
    """Return the five per-class accuracies and balanced accuracy."""
    class_accuracies = []

    for k in range(confusion_matrix.shape[0]):
        correct = confusion_matrix[k, k].item()
        total = confusion_matrix[k].sum().item()
        class_accuracies.append(correct / total if total else 0.0)

    balanced = sum(class_accuracies) / len(class_accuracies)
    return class_accuracies, balanced
