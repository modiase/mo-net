import sys
import time
from pathlib import Path
from typing import Final

import click
import pandas as pd
import plotille
from loguru import logger

from mnist_numpy.data import DATA_DIR

DEFAULT_REFRESH_SECONDS: Final[int] = 1
DEFAULT_WIDTH: Final[int] = 150
DEFAULT_HEIGHT: Final[int] = 40


@click.command()
@click.option(
    "--training_log_path", "-t", type=Path, help="Path to the training log file"
)
@click.option(
    "--refresh",
    "-r",
    type=int,
    default=DEFAULT_REFRESH_SECONDS,
    help="Refresh interval in seconds",
)
@click.option(
    "--width", "-w", type=int, default=DEFAULT_WIDTH, help="Plot width in characters"
)
@click.option(
    "--height", "-h", type=int, default=DEFAULT_HEIGHT, help="Plot height in characters"
)
def main(
    *,
    training_log_path: Path | None = None,
    refresh: int,
    width: int,
    height: int,
):
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

    logger.info(f"Monitoring {training_log_path} (refresh: {refresh}s)")

    last_modified = 0.0
    last_epoch = -1

    try:
        while True:
            current_modified = training_log_path.stat().st_mtime

            if current_modified > last_modified:
                last_modified = current_modified

                df = pd.read_csv(training_log_path)

                if df.empty or (
                    not df.empty and float(df["epoch"].max()) == last_epoch
                ):
                    time.sleep(refresh)
                    continue

                last_epoch = int(df["epoch"].max())

                print("\033c", end="")

                fig = plotille.Figure()
                fig.width = width
                fig.height = height

                x_max = float(max(float(df["epoch"].max()) + 1, 10))
                fig.set_x_limits(min_=0.0, max_=x_max)

                min_y = float(
                    min(
                        df["batch_loss"].min(),
                        df["val_loss"].min(),
                        df["monotonic_val_loss"].min(),
                    )
                    * 0.95
                )

                max_y = float(
                    max(
                        df["batch_loss"].max(),
                        df["val_loss"].max(),
                        df["monotonic_val_loss"].max(),
                    )
                    * 1.05
                )

                fig.set_y_limits(min_=min_y, max_=max_y)

                epochs = df["epoch"].tolist()

                fig.plot(epochs, df["val_loss"].tolist(), label="Validation Loss")
                fig.plot(
                    epochs,
                    df["monotonic_val_loss"].tolist(),
                    label="Monotonic Validation Loss",
                )

                fig.x_label = "Epoch"
                fig.y_label = "Loss"
                fig.title = f"Training Progress (Epoch {int(last_epoch)})"

                print(fig.show())

                latest = df.iloc[-1]
                print(f"Epoch: {int(latest['epoch'])}")
                print(f"Batch Loss: {float(latest['batch_loss']):.6f}")
                print(f"Validation Loss: {float(latest['val_loss']):.6f}")
                print(f"Learning Rate: {float(latest['learning_rate']):.8f}")
                print(f"Last Update: {latest['timestamp']}")

            time.sleep(refresh)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
