from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd

from mnist_numpy.data import DATA_DIR


@click.command()
@click.argument("training_log_path", type=Path)
def main(training_log_path: Path):
    if not training_log_path.exists():
        training_log_path = DATA_DIR / training_log_path.name
    # Read the CSV file
    df = pd.read_csv(training_log_path)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["epoch"], df["training_loss"], label="Training Loss", color="blue")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss", color="red")
    plt.plot(
        df["epoch"],
        df["monotonic_training_loss"],
        label="Monotonic Training Loss",
        color="orange",
    )
    plt.plot(
        df["epoch"],
        df["monotonic_test_loss"],
        label="Monotonic Test Loss",
        color="green",
    )

    # Customize the plot
    plt.title("Training and Test Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Use scientific notation for y-axis when numbers are very small
    plt.yscale("log")

    # Show the plot
    plt.tight_layout()
    plt.show()
