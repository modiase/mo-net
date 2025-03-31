from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd


@click.command()
@click.argument("training_log_path", type=click.Path(exists=True))
def main(training_log_path: Path):
    # Read the CSV file
    df = pd.read_csv(training_log_path)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["epoch"], df["training_loss"], label="Training Loss", color="blue")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss", color="red")

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
