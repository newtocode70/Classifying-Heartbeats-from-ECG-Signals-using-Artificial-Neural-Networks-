import os
import time
import torch
import matplotlib.pyplot as plt
from src.prepare import prepare_data
from src.model import ECGNet, cross_entropy

def train(alpha, X_train_tensor, y_train_tensor,
          X_val_tensor, y_val_tensor):
    torch.manual_seed(0)
    model = ECGNet()

    batch_size = 128
    beta = 0.9
    weight_decay = 1e-4
    epochs = 30

    velocities = {}
    for name, param in model.named_parameters():
        velocities[name] = torch.zeros_like(param)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(X_train_tensor))

        for start in range(0, len(X_train_tensor), batch_size):
            indices = permutation[start:start + batch_size]
            xb = X_train_tensor[indices]
            yb = y_train_tensor[indices]

            model.zero_grad()
            logits = model(xb)
            loss = cross_entropy(logits, yb)
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    gradient = param.grad

                    # Weight decay is applied to weights, not biases.
                    if name.startswith("W"):
                        gradient = gradient + weight_decay * param

                    velocities[name].mul_(beta).add_(
                        gradient, alpha=(1 - beta)
                    )
                    param.sub_(alpha * velocities[name])

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train_tensor)
            train_loss = cross_entropy(
                train_logits, y_train_tensor
            ).item()
            train_predictions = torch.argmax(train_logits, dim=1)
            train_accuracy = (
                train_predictions == y_train_tensor
            ).float().mean().item()

            val_logits = model(X_val_tensor)
            val_loss = cross_entropy(
                val_logits, y_val_tensor
            ).item()
            val_predictions = torch.argmax(val_logits, dim=1)
            val_accuracy = (
                val_predictions == y_val_tensor
            ).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                name: param.detach().clone()
                for name, param in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    return model, history, best_state, best_val_loss, best_epoch


def main():
    os.makedirs("results", exist_ok=True)
    (X_train, y_train, X_val, y_val,
     X_test, y_test) = prepare_data()

    learning_rates = [0.01, 0.1, 0.5]
    results_by_alpha = {}
    total_start = time.time()

    for alpha in learning_rates:
        print("\nTraining with learning rate:", alpha)
        model, history, best_state, best_val_loss, best_epoch = train(
            alpha, X_train, y_train, X_val, y_val
        )
        results_by_alpha[alpha] = {
            "model": model,
            "history": history,
            "best_state": best_state,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch
        }
        print("Best validation loss:", best_val_loss)
        print("Best epoch:", best_epoch)

    training_seconds = time.time() - total_start

    print("\nLearning Rate Comparison")
    for alpha in learning_rates:
        item = results_by_alpha[alpha]
        best_val_accuracy = item["history"]["val_accuracy"][item["best_epoch"] - 1]
        print(
            f"LR: {alpha} | Lowest validation loss: "
            f"{item['best_val_loss']:.6f} | "
            f"Validation accuracy at best epoch: {best_val_accuracy:.4f}"
        )

    plt.figure(figsize=(10, 6))
    for alpha in learning_rates:
        val_losses = results_by_alpha[alpha]["history"]["val_loss"]
        plt.plot(range(1, 31), val_losses,
                 label=f"Learning Rate = {alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/learning_rate_comparison.png", dpi=150)
    plt.close()

    chosen_alpha = min(
        learning_rates,
        key=lambda a: results_by_alpha[a]["best_val_loss"]
    )
    selected = results_by_alpha[chosen_alpha]
    selected_model = selected["model"]
    selected_model.load_state_dict(selected["best_state"])
    best_history = selected["history"]
    best_epoch = selected["best_epoch"]

    print("\nSelected Learning Rate:", chosen_alpha)
    print("Lowest Validation Loss:", selected["best_val_loss"])
    print("Best Epoch:", best_epoch)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 31), best_history["train_loss"], label="Training Loss")
    plt.plot(range(1, 31), best_history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss (LR = {chosen_alpha})")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/training_validation_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 31), best_history["train_accuracy"],
             label="Training Accuracy")
    plt.plot(range(1, 31), best_history["val_accuracy"],
             label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy (LR = {chosen_alpha})")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/training_validation_accuracy.png", dpi=150)
    plt.close()

    torch.save(selected_model.state_dict(), "results/best_model.pth")

    # Save a compact record of the selected run and training time.
    import pandas as pd
    summary_rows = []
    for alpha in learning_rates:
        item = results_by_alpha[alpha]
        ep = item["best_epoch"] - 1
        summary_rows.append({
            "learning_rate": alpha,
            "lowest_validation_loss": item["best_val_loss"],
            "validation_accuracy_at_best_epoch":
                item["history"]["val_accuracy"][ep],
            "best_epoch": item["best_epoch"]
        })
    pd.DataFrame(summary_rows).to_csv(
        "results/learning_rate_results.csv", index=False
    )

    print("\nBest model saved: results/best_model.pth")
    print(f"Total training time: {training_seconds:.2f} seconds")
    print("Hardware/device: CPU")


if __name__ == "__main__":
    main()
