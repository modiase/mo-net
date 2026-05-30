import sys
import time
from typing import Final

import click
import pandas as pd
import plotille
from InquirerPy import inquirer
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from mo_net.settings import get_settings
from mo_net.train.backends.models import DbRun

DEFAULT_REFRESH_SECONDS: Final[int] = 1
DEFAULT_WIDTH: Final[int] = 150
DEFAULT_HEIGHT: Final[int] = 40


def get_run_data(session, run_id: int) -> pd.DataFrame | None:
    """Pivot the EAV `metrics` table into per-step rows for ``run_id``."""
    rows = session.execute(
        text(
            """
            SELECT i.step AS batch, i.epoch, i.timestamp, m.name, m.value
            FROM iterations i
            JOIN metrics m USING (run_id, step)
            WHERE i.run_id = :run_id
            ORDER BY i.step
            """
        ),
        {"run_id": run_id},
    ).fetchall()
    if not rows:
        return None

    long_df = pd.DataFrame(
        rows, columns=["batch", "epoch", "timestamp", "name", "value"]
    )
    wide_df = long_df.pivot_table(
        index=["batch", "epoch", "timestamp"],
        columns="name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide_df.columns.name = None
    return wide_df.sort_values("batch").reset_index(drop=True)


def get_available_runs(session) -> list[DbRun]:
    return session.query(DbRun).order_by(DbRun.started_at.desc()).all()


def prompt_for_run_selection(session) -> int:
    runs = get_available_runs(session)
    if not runs:
        logger.error("No runs found in database")
        sys.exit(1)

    choices = []
    for run in runs:
        last_ts = run.last_iteration_at or run.started_at
        display_name = f"Run {run.id} (Started: {run.started_at}, Last: {last_ts})"
        choices.append({"name": display_name, "value": run.id})

    run_id = inquirer.select(  # type: ignore[attr-defined]
        message="Select a training run to monitor:",
        choices=choices,
    ).execute()

    if run_id is None:
        sys.exit(1)

    return run_id


@click.command()
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
def main(*, refresh: int, width: int, height: int):
    db_path = get_settings().resolved_db_path
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()
    logger.info(f"Using database: {db_path}")

    run_id = prompt_for_run_selection(session)
    logger.info(f"Monitoring run {run_id} (refresh: {refresh}s)")

    last_epoch = -1
    last_row_count = 0

    try:
        while True:
            df = get_run_data(session, run_id)
            if df is None or df.empty:
                time.sleep(refresh)
                continue

            current_row_count = len(df)
            data_updated = current_row_count > last_row_count
            last_row_count = current_row_count

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
            fig.title = f"Training Progress (Epoch {int(last_epoch)})"  # type: ignore[attr-defined]

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
        session.close()


if __name__ == "__main__":
    main()
