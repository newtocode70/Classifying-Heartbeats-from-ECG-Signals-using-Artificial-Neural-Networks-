import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.prepare import prepare_data
from src.model import ECGNet
from src.metrics import accuracy, make_confusion_matrix, per_class_accuracy

def main():
    os.makedirs("results", exist_ok=True)

    (_, _, _, _, X_test_tensor, y_test_tensor) = prepare_data()

    test_model = ECGNet()
    state = torch.load(
        "results/best_model.pth", map_location="cpu"
    )
    test_model.load_state_dict(state)
    test_model.eval()

    with torch.no_grad():
        test_logits = test_model(X_test_tensor)
        test_predictions = torch.argmax(test_logits, dim=1)

    test_accuracy = accuracy(y_test_tensor, test_predictions)
    confusion_matrix = make_confusion_matrix(
        y_test_tensor, test_predictions, 5
    )
    class_accuracies, balanced_accuracy = per_class_accuracy(
        confusion_matrix
    )

    print("Test predictions generated:", len(test_predictions))
    print("Correct predictions:",
          (test_predictions == y_test_tensor).sum().item())
    print("Total predictions:", len(y_test_tensor))
    print("Test accuracy:", test_accuracy)
    print("Test accuracy (%):", test_accuracy * 100)

    print("\nConfusion Matrix (rows=true, columns=predicted):")
    print(confusion_matrix)
    print("Confusion matrix total:", confusion_matrix.sum().item())

    class_names = [
        "N: Normal",
        "S: Supraventricular",
        "V: Ventricular",
        "F: Fusion",
        "Q: Unknown/Paced"
    ]

    print("\nPer-class accuracy:")
    for name, value in zip(class_names, class_accuracies):
        print(name, ":", round(value * 100, 2), "%")

    print("\nBalanced accuracy:", balanced_accuracy)
    print("Balanced accuracy (%):", balanced_accuracy * 100)

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix.numpy(), cmap="Blues")
    plt.title("ECG Heartbeat Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(range(5), ["N", "S", "V", "F", "Q"])
    plt.yticks(range(5), ["N", "S", "V", "F", "Q"])
    plt.colorbar()

    for i in range(5):
        for j in range(5):
            plt.text(
                j, i, str(confusion_matrix[i, j].item()),
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()

    cm_df = pd.DataFrame(
        confusion_matrix.numpy(),
        index=["N", "S", "V", "F", "Q"],
        columns=["N", "S", "V", "F", "Q"]
    )
    cm_df.to_csv("results/confusion_matrix.csv")

    results_df = pd.DataFrame({
        "Class": class_names,
        "Accuracy (%)": [value * 100 for value in class_accuracies]
    })
    results_df.to_csv(
        "results/per_class_accuracy.csv", index=False
    )

    pd.DataFrame([{
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_percent": balanced_accuracy * 100
    }]).to_csv("results/test_summary.csv", index=False)

    print("\nTask 5 test results saved in results/.")


if __name__ == "__main__":
    main()
