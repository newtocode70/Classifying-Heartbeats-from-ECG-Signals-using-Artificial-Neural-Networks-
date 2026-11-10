import numpy as np
import torch

class ECGNet(torch.nn.Module):
    """187 -> 128 ReLU -> 64 ReLU -> 5 logits."""

    def __init__(self):
        super(ECGNet, self).__init__()

        self.W1 = torch.nn.Parameter(
            torch.randn(128, 187) * np.sqrt(2 / 187)
        )
        self.b1 = torch.nn.Parameter(torch.zeros(128))

        self.W2 = torch.nn.Parameter(
            torch.randn(64, 128) * np.sqrt(2 / 128)
        )
        self.b2 = torch.nn.Parameter(torch.zeros(64))

        self.W3 = torch.nn.Parameter(
            torch.randn(5, 64) * np.sqrt(2 / 64)
        )
        self.b3 = torch.nn.Parameter(torch.zeros(5))

    def forward(self, x):
        z1 = x @ self.W1.T + self.b1
        a1 = torch.clamp(z1, min=0)

        z2 = a1 @ self.W2.T + self.b2
        a2 = torch.clamp(z2, min=0)

        z3 = a2 @ self.W3.T + self.b3
        return z3


def cross_entropy(logits, y):
    """Mean softmax cross-entropy implemented using logsumexp."""
    batch_size = logits.shape[0]
    correct_class_logits = logits[torch.arange(batch_size), y]
    loss = -correct_class_logits + torch.logsumexp(logits, dim=1)
    return loss.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ECGNet()
    total_parameters = sum(p.numel() for p in model.parameters())
    print("ECG Neural Network created successfully!")
    print("Total trainable parameters:", total_parameters)
