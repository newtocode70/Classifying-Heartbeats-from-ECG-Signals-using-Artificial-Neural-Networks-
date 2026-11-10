import numpy as np
import torch
from src.load_data import load_raw

def prepare_data():
    """Separate signals/labels, create the required split, and return tensors."""
    raw = load_raw()
    train_df = raw["train"]
    test_df = raw["test"]

    X_train = train_df.iloc[:, 0:187]
    y_train = train_df.iloc[:, 187].astype(int)

    X_test = test_df.iloc[:, 0:187]
    y_test = test_df.iloc[:, 187].astype(int)

    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)
    print("Training input shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing input shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)

    print("Minimum signal value:", X_train.min().min())
    print("Maximum signal value:", X_train.max().max())

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    class_names = [
        "N - Normal",
        "S - Supraventricular",
        "V - Ventricular",
        "F - Fusion",
        "Q - Unknown/Paced"
    ]

    print("\nClass counts:")
    for i in range(5):
        print(class_names[i])
        print("Training:", train_counts.get(i, 0))
        print("Testing:", test_counts.get(i, 0))

    # Exact assignment split: 10% validation, random seed 42.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    print("\nTraining rows:", len(train_idx))
    print("Validation rows:", len(val_idx))
    print("Testing rows:", len(test_df))

    X_train_tensor = torch.tensor(
        X_train.iloc[train_idx].values, dtype=torch.float32
    )
    y_train_tensor = torch.tensor(
        y_train.iloc[train_idx].values, dtype=torch.long
    )
    X_val_tensor = torch.tensor(
        X_train.iloc[val_idx].values, dtype=torch.float32
    )
    y_val_tensor = torch.tensor(
        y_train.iloc[val_idx].values, dtype=torch.long
    )
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    print("\nTensor shapes:")
    print("Train:", X_train_tensor.shape, y_train_tensor.shape)
    print("Validation:", X_val_tensor.shape, y_val_tensor.shape)
    print("Test:", X_test_tensor.shape, y_test_tensor.shape)

    return (
        X_train_tensor, y_train_tensor,
        X_val_tensor, y_val_tensor,
        X_test_tensor, y_test_tensor
    )


if __name__ == "__main__":
    prepare_data()
