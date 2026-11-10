import torch
from src.prepare import prepare_data
from src.model import ECGNet, cross_entropy

def run_gradient_check():
    X_train, y_train, _, _, _, _ = prepare_data()

    torch.manual_seed(0)
    model = ECGNet()

    X_batch = X_train[:64]
    y_batch = y_train[:64]

    model.zero_grad()
    logits = model(X_batch)
    loss = cross_entropy(logits, y_batch)
    loss.backward()

    z1 = X_batch @ model.W1.T + model.b1
    a1 = torch.clamp(z1, min=0)
    z2 = a1 @ model.W2.T + model.b2
    a2 = torch.clamp(z2, min=0)

    Y_hat = torch.softmax(logits.detach(), dim=1)
    Y = torch.zeros(64, 5)
    Y[torch.arange(64), y_batch] = 1

    B = 64
    grad_W3_manual = (Y_hat - Y).T @ a2.detach() / B
    grad_b3_manual = (Y_hat - Y).sum(dim=0) / B

    W3_difference = torch.max(
        torch.abs(grad_W3_manual - model.W3.grad)
    )
    b3_difference = torch.max(
        torch.abs(grad_b3_manual - model.b3.grad)
    )

    print("Loss:", loss.item())
    print("Manual W3 gradient shape:", grad_W3_manual.shape)
    print("Manual b3 gradient shape:", grad_b3_manual.shape)
    print("Maximum difference in W3 gradients:", W3_difference.item())
    print("Maximum difference in b3 gradients:", b3_difference.item())

    if W3_difference.item() < 1e-5 and b3_difference.item() < 1e-5:
        print("Gradient check passed.")
    else:
        print("Gradient check failed. Check the calculations.")


if __name__ == "__main__":
    run_gradient_check()
