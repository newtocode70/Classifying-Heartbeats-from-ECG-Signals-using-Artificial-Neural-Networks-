import os
import matplotlib.pyplot as plt
from src.load_data import load_raw

CLASS_NAMES = [
    "N: normal",
    "S: supraventricular ectopic",
    "V: ventricular ectopic",
    "F: fusion",
    "Q: unknown / paced"
]

FS = 125

def plot_one_per_class(df, out_path="results/example_beats.png"):
    """Plot the first heartbeat example for each of the five classes."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    time_ms = [1000 * i / FS for i in range(187)]

    for k, ax in enumerate(axes):
        class_rows = df[df[187] == k]

        if class_rows.empty:
            raise ValueError(f"No example found for class {k}.")

        row = class_rows.iloc[0]
        ax.plot(time_ms, row.iloc[:187].values)
        ax.set_title(CLASS_NAMES[k], fontsize=9)
        ax.set_xlabel("time (ms)")

    axes[0].set_ylabel("normalised amplitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    plt.show()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    raw = load_raw()
    plot_one_per_class(raw["train"])
