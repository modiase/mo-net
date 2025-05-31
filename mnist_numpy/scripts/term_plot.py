import sys
import time
from pathlib import Path
from typing import Final

import click
import pandas as pd
import plotille
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mnist_numpy.data import DATA_DIR
from mnist_numpy.train.server.models import DB_PATH, DbRun, Iteration

DEFAULT_REFRESH_SECONDS: Final[int] = 1
DEFAULT_WIDTH: Final[int] = 150
DEFAULT_HEIGHT: Final[int] = 40


def get_latest_run_data(session) -> pd.DataFrame | None:
    latest_run = session.query(DbRun).order_by(DbRun.updated_at.desc()).first()
    if not latest_run:
        return None

    iterations = (
        session.query(Iteration)
        .filter(Iteration.run_id == latest_run.id)
        .order_by(Iteration.timestamp)
        .all()
    )
    if not iterations:
        return None

    return pd.DataFrame(
        [
            {
                "batch_loss": it.batch_loss,
                "val_loss": it.val_loss,
                "batch": it.batch,
                "epoch": it.epoch,
                "learning_rate": it.learning_rate,
                "timestamp": it.timestamp,
            }
            for it in iterations
        ]
    )


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
    *, training_log_path: Path | None = None, refresh: int, width: int, height: int
):
    use_database = training_log_path is None

    if use_database:
        if not DB_PATH.exists():
            logger.error(f"Database not found: {DB_PATH}")
            sys.exit(1)
        engine = create_engine(f"sqlite:///{DB_PATH}")
        Session = sessionmaker(bind=engine)
        session = Session()
        logger.info(f"Using database: {DB_PATH}")
    else:
        if training_log_path is None:
            run_dir = DATA_DIR / "run"
            training_log_files = tuple(run_dir.glob("*_training_log.csv"))
            if not training_log_files:
                logger.error(f"No training log files found in {run_dir}")
                sys.exit(1)
            training_log_path = max(training_log_files, key=lambda p: p.stat().st_mtime)

        if not training_log_path.exists():
            logger.error(f"File not found: {training_log_path}")
            sys.exit(1)
        logger.info(f"Using CSV file: {training_log_path}")

    logger.info(f"Monitoring (refresh: {refresh}s)")

    last_modified = 0.0
    last_epoch = -1
    last_row_count = 0

    try:
        while True:
            if use_database:
                df = get_latest_run_data(session)
                if df is None or df.empty:
                    time.sleep(refresh)
                    continue
                current_row_count = len(df)
                data_updated = current_row_count > last_row_count
                last_row_count = current_row_count
            else:
                current_modified = training_log_path.stat().st_mtime
                data_updated = current_modified > last_modified
                if data_updated:
                    last_modified = current_modified
                    df = pd.read_csv(training_log_path)

            if (
                not data_updated
                or df.empty
                or (not df.empty and float(df["epoch"].max()) == last_epoch)
            ):
                time.sleep(refresh)
                continue

            last_epoch = int(df["epoch"].max())
            monotonic_val_loss = df["val_loss"].cummin().tolist()

            print("\033c", end="")

            fig = plotille.Figure()
            fig.width = width
            fig.height = height

            x_max = float(max(float(df["epoch"].max()) + 1, 10))
            fig.set_x_limits(min_=0.0, max_=x_max)

            min_y = float(min(df["batch_loss"].min(), df["val_loss"].min()) * 0.95)
            max_y = float(max(df["batch_loss"].max(), df["val_loss"].max()) * 1.05)
            fig.set_y_limits(min_=min_y, max_=max_y)

            epochs = df["epoch"].tolist()
            fig.plot(epochs, df["val_loss"].tolist(), label="Validation Loss")
            fig.plot(epochs, monotonic_val_loss, label="Monotonic Validation Loss")

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
    finally:
        if use_database:
            session.close()


if __name__ == "__main__":
    main()
