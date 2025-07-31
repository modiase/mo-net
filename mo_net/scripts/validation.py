import asyncio
import functools
import sys
import tempfile
from contextlib import asynccontextmanager, contextmanager
from math import ceil
from pathlib import Path
from urllib.parse import urlparse

import click
import jax.numpy as jnp
import pandas as pd
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

from mo_net.cli import train, training_options
from mo_net.resources import get_resource
from mo_net.train.trainer.trainer import TrainingResult


def _make_logging_backend_connection_string(tmp_dir: Path, split_index: int) -> str:
    return f"csv://{tmp_dir / f'fold_{split_index}.csv'}"


def _get_current_epoch(split_index: int, tmp_dir: Path) -> int:
    """Get the current epoch for a given fold."""
    try:
        df = pd.read_csv(
            urlparse(_make_logging_backend_connection_string(tmp_dir, split_index)).path
        )
        if df.empty:
            return 0
        max_epoch = df["epoch"].max()
        return 0 if pd.isna(max_epoch) else int(max_epoch)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0


def _get_min_val_loss(split_index: int, tmp_dir: Path) -> float:
    df = pd.read_csv(
        urlparse(_make_logging_backend_connection_string(tmp_dir, split_index)).path
    )
    return df["val_loss"].min()


def _validate_kwargs(**kwargs) -> None:
    if kwargs.pop("workers") != 0:
        raise ValueError("workers is not supported when running cross-validation")
    if kwargs.get("train_split_index") != 0:
        raise ValueError(
            "train_split_index is not supported when running cross-validation"
        )


def _run_fold(split_index: int, *, tmp_dir: Path, **kwargs) -> TrainingResult | None:
    """Helper function to run a single training fold."""
    kwargs["logging_backend_connection_string"] = (
        _make_logging_backend_connection_string(tmp_dir, split_index)
    )
    kwargs["model_output_path"] = tmp_dir / f"model_{split_index}.pkl"
    kwargs["no_monitoring"] = True
    kwargs["quiet"] = True
    kwargs["seed"] = split_index
    kwargs["train_split_index"] = split_index
    kwargs["workers"] = 0
    try:
        return train(**kwargs)
    except Exception as e:
        logger.error(f"Fold {split_index} failed: {e}")
        return None


async def _run_fold_async(
    split_index: int, *, tmp_dir: Path, **kwargs
) -> TrainingResult | None:
    """Async wrapper for running a single training fold."""
    return await asyncio.get_event_loop().run_in_executor(
        None, functools.partial(_run_fold, split_index, tmp_dir=tmp_dir, **kwargs)
    )


async def _monitor_progress(
    tmp_dir: Path, n_splits: int, num_epochs: int, pbar: tqdm
) -> None:
    """Monitor progress of all folds and update the progress bar."""
    while True:
        try:
            current_epochs = [
                _get_current_epoch(split_index, tmp_dir)
                for split_index in range(n_splits)
            ]
            total_epochs_completed = sum(current_epochs)
            pbar.n = total_epochs_completed
            pbar.refresh()

            if total_epochs_completed >= num_epochs * n_splits:
                break

            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error monitoring progress: {e}")
            await asyncio.sleep(1)


@contextmanager
def filter_logs():
    """Context manager to filter logs to only show those with name == 'leader'."""
    logger.remove()
    logger.add(
        sys.stdout,
        filter=lambda record: record["name"] == "leader",
        level="INFO",
    )
    try:
        yield
    finally:
        logger.remove()
        logger.add(sys.stdout, level="INFO")


@asynccontextmanager
async def monitor_progress(tmp_dir: Path, n_splits: int, num_epochs: int, pbar: tqdm):
    """Context manager to handle the progress monitoring task lifecycle."""
    monitor_task = asyncio.create_task(
        _monitor_progress(tmp_dir, n_splits, num_epochs, pbar)
    )
    try:
        yield monitor_task
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


async def validate(**kwargs) -> None:
    """
    Run cross-validation by training the model on multiple splits of the data in
    parallel using async processing.
    """
    _validate_kwargs(**kwargs)

    n_splits = ceil(1 / (1 - kwargs.get("train_split", 0.8)))
    num_epochs = kwargs["num_epochs"]
    logger.info(
        f"Running cross-validation with {n_splits} splits and {num_epochs} epochs."
    )

    if (dataset_url := kwargs.get("dataset_url")) is None:
        raise ValueError("dataset_url is required")
    get_resource(dataset_url)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        logger.info(f"Using temporary directory: {Path(tmp_dir_name)}")

        with (
            filter_logs(),
            tqdm(
                total=num_epochs * n_splits,
                desc="Cross-validation Training Progress",
                unit="epoch",
            ) as pbar,
        ):
            async with monitor_progress(
                Path(tmp_dir_name),
                n_splits,
                num_epochs,
                pbar,
            ):
                results = await asyncio.gather(
                    *[
                        _run_fold_async(
                            split_index, tmp_dir=Path(tmp_dir_name), **kwargs
                        )
                        for split_index in range(n_splits)
                    ]
                )

                logger.info(
                    f"Cross-validation finished. {len([r for r in enumerate(results) if r is not None])}/{n_splits} runs were successful."
                )

                min_val_losses = jnp.array(
                    [
                        _get_min_val_loss(split_index, Path(tmp_dir_name))
                        for split_index in range(n_splits)
                    ]
                )

    logger.info(
        "\nValidation Loss Statistics:"
        + "\n"
        + tabulate(
            [
                [k, f"{v:.6f}"]
                for k, v in {
                    "Min": jnp.min(min_val_losses),
                    "Max": jnp.max(min_val_losses),
                    "Mean": jnp.mean(min_val_losses),
                    "Std Dev": jnp.std(min_val_losses),
                }.items()
            ],
            headers=["Metric", "Value"],
            tablefmt="grid",
        )
    )

    logger.info(
        "\nValidation Runs:"
        + "\n"
        + tabulate(
            [[i, f"{loss:.6f}"] for i, loss in enumerate(min_val_losses)],
            headers=["Split Index", "Min Val Loss"],
            tablefmt="grid",
        )
    )


@click.command(help="Run cross-validation")
@training_options
@click.option(
    "-o",
    "--optimiser-type",
    type=click.Choice(["adam", "none", "rmsprop"]),
    help="The type of optimiser to use",
    default="adam",
)
def main(**kwargs) -> None:
    """Synchronous wrapper for the async validate function."""
    asyncio.run(validate(**kwargs))


if __name__ == "__main__":
    main()
