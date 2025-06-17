from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Final, Self, overload

import numpy as np
import pandas as pd
from loguru import logger

from mo_net.log import LogLevel, log_result
from mo_net.resources import MNIST_TRAIN_URL, get_resource

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEFAULT_TRAIN_SPLIT: Final[float] = 0.8
MAX_PIXEL_VALUE: Final[int] = 255
N_DIGITS: Final[int] = 10
OUTPUT_PATH: Final[Path] = DATA_DIR / "output"
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RUN_PATH: Final[Path] = DATA_DIR / "run"
if not RUN_PATH.exists():
    RUN_PATH.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, kw_only=True)
class SplitConfig:
    split_ratio: float
    split_index: int

    @property
    def val_ratio(self) -> float:
        return 1 - self.split_ratio

    @property
    def n_splits(self) -> int:
        return ceil(1 / self.val_ratio)

    @classmethod
    def of(cls, split_ratio: float, split_index: int = 0) -> Self:
        return cls(split_ratio=split_ratio, split_index=split_index)

    def __post_init__(self):
        if self.split_index >= self.n_splits:
            raise ValueError(
                f"Split index {self.split_index} must be less than the number of splits {self.n_splits}"
            )

    @log_result(LogLevel.DEBUG)
    def get_train_indices(self, train_set_size: int) -> Sequence[tuple[int, int]]:
        val_size = int(train_set_size * self.val_ratio)
        val_start = self.split_index * val_size
        val_end = val_start + val_size

        if val_start == 0:
            return ((val_end, train_set_size),)
        elif val_end >= train_set_size:
            return ((0, val_start),)
        else:
            return ((0, val_start), (val_end, train_set_size))

    @log_result(LogLevel.DEBUG)
    def get_val_indices(self, train_set_size: int) -> tuple[int, int]:
        val_size = int(train_set_size * (1 - self.split_ratio))
        val_start = self.split_index * val_size
        val_end = min(val_start + val_size, train_set_size)
        return val_start, val_end


def _load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    return (
        np.array(df.iloc[:, 1:]) / MAX_PIXEL_VALUE,
        np.eye(N_DIGITS)[df.iloc[:, 0].to_numpy()],
    )


def _load_data_split(
    data_path: Path, split: SplitConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)

    train_indices = split.get_train_indices(len(df))
    val_start, val_end = split.get_val_indices(len(df))

    val_set = df.iloc[val_start:val_end, :]
    training_set = pd.concat(
        [df.iloc[start:end, :] for start, end in train_indices], ignore_index=True
    )

    Y_train = np.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
    Y_val = np.eye(N_DIGITS)[val_set.iloc[:, 0].to_numpy()]

    X_train = np.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_val = np.array(val_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_val, Y_val


@overload
def load_data(
    dataset_url: str, split: None = None
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def load_data(
    dataset_url: str, split: SplitConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def load_data(
    dataset_url: str, split: SplitConfig | None = None
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray]
):
    logger.info(f"Loading data from {dataset_url}.")
    data_path = get_resource(dataset_url)
    return _load_data_split(data_path, split) if split else _load_data(data_path)


def infer_dataset_url(quickstart: str | None) -> str | None:
    if quickstart == "mnist_mlp" or quickstart == "mnist_cnn":
        return MNIST_TRAIN_URL
    return None
