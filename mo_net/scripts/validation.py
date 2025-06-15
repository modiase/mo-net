import functools
import multiprocessing
import os
import sys
import tempfile
from math import ceil
from pathlib import Path
from urllib.parse import urlparse

import click
import pandas as pd
from loguru import logger

from mo_net.cli import train, training_options
from mo_net.resources import get_resource
from mo_net.train.trainer.trainer import TrainingResult


def _make_logging_backend_connection_string(tmp_dir: Path, split_index: int) -> str:
    return f"csv://{tmp_dir / f'fold_{split_index}.csv'}"


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


def _validate_kwargs(**kwargs) -> None:
    if kwargs.pop("workers") != 0:
        raise ValueError("workers is not supported when running cross-validation")
    if kwargs.get("train_split_index") != 0:
        raise ValueError(
            "train_split_index is not supported when running cross-validation"
        )


def _get_min_val_loss(split_index: int, tmp_dir: Path) -> float:
    df = pd.read_csv(
        urlparse(_make_logging_backend_connection_string(tmp_dir, split_index)).path
    )
    return df["val_loss"].min()


@click.command(help="Run cross-validation")
@training_options
def validate(**kwargs) -> None:
    """
    Run cross-validation by training the model on multiple splits of the data in
    parallel.
    """
    _validate_kwargs(**kwargs)

    train_split = kwargs.get("train_split", 0.8)
    kwargs.pop("train_split_index")

    n_splits = ceil(1 / (1 - train_split))
    logger.info(f"Running cross-validation with {n_splits} splits.")
    dataset_url = kwargs.get("dataset_url")
    if dataset_url is None:
        raise ValueError("dataset_url is required")
    get_resource(dataset_url)  # Pre-fetch the dataset

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        partial_run_fold = functools.partial(_run_fold, tmp_dir=tmp_dir, **kwargs)
        logger.info(f"Using temporary directory: {tmp_dir}")
        logger.remove()
        logger.add(
            sys.stdout,
            filter=lambda record: record["name"] != "trainer",
            level="INFO",
        )
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = pool.map(partial_run_fold, range(n_splits))

        successful_runs = [r for r in enumerate(results) if r is not None]
        min_val_losses = [
            _get_min_val_loss(split_index, tmp_dir) for split_index in range(n_splits)
        ]
    logger.info(
        f"Cross-validation finished. {len(successful_runs)}/{n_splits} runs were successful."
    )
    logger.info(f"Results: {successful_runs}")
    logger.info(f"Min val losses: {min_val_losses}")


if __name__ == "__main__":
    validate()
