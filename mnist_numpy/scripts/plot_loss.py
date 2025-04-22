import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from mnist_numpy.data import DATA_DIR


@click.command()
@click.option(
    "--training_log_path", "-t", type=Path, help="Path to the training log file"
)
def main(training_log_path: Path | None = None):
    if training_log_path is None:
        run_dir = DATA_DIR / "run"
        training_log_files = tuple(run_dir.glob("*_training_log.csv"))
        if not training_log_files:
            logger.error(f"No training log files found in {run_dir}")
            sys.exit(1)
        training_log_path = max(training_log_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest training log file: {training_log_path}")
    if not training_log_path.exists():
        logger.error(f"File not found: {training_log_path}")
        sys.exit(1)
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
