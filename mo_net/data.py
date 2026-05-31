from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Final, Self, overload

import jax.numpy as jnp
import pandas as pd
from loguru import logger

from mo_net.log import LogLevel, log_result
from mo_net.resources import get_resource
from mo_net.settings import get_settings

DEFAULT_TRAIN_SPLIT: Final[float] = 0.8
MAX_PIXEL_VALUE: Final[int] = 255
N_DIGITS: Final[int] = 10


def __getattr__(name: str):
    # Back-compat shim: callers that did `from mo_net.data import DATA_DIR`
    # continue to work, but new code should call `get_settings()` directly so
    # CLI/env overrides take effect at runtime.
    settings = get_settings()
    if name == "DATA_DIR":
        return settings.data_dir
    if name == "OUTPUT_PATH":
        return settings.output_dir
    if name == "RUN_PATH":
        return settings.run_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _load_data(df: pd.DataFrame, *, one_hot: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (
        jnp.array(df.iloc[:, 1:]) / MAX_PIXEL_VALUE,
        (
            jnp.eye(N_DIGITS)[df.iloc[:, 0].to_numpy()]
            if one_hot
            else jnp.array(df.iloc[:, 0].to_numpy())
        ),
    )


def _load_data_split(
    df: pd.DataFrame, split: SplitConfig, *, one_hot: bool
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    train_indices = split.get_train_indices(len(df))
    val_start, val_end = split.get_val_indices(len(df))

    val_set = df.iloc[val_start:val_end, :]
    training_set = pd.concat(
        [df.iloc[start:end, :] for start, end in train_indices], ignore_index=True
    )

    Y_train = (
        jnp.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
        if one_hot
        else jnp.array(training_set.iloc[:, 0].to_numpy())
    )
    Y_val = (
        jnp.eye(N_DIGITS)[val_set.iloc[:, 0].to_numpy()]
        if one_hot
        else jnp.array(val_set.iloc[:, 0].to_numpy())
    )

    X_train = jnp.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_val = jnp.array(val_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_val, Y_val


@overload
def load_data(
    dataset_url: str,
    split: None = None,
    *,
    one_hot: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]: ...
@overload
def load_data(
    dataset_url: str,
    split: SplitConfig,
    *,
    one_hot: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: ...


def load_data(
    dataset_url: str,
    split: SplitConfig | None = None,
    *,
    one_hot: bool = True,
) -> (
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray]
):
    logger.info(f"Loading data from {dataset_url}.")
    df = get_resource(dataset_url).to_pandas()
    assert isinstance(df, pd.DataFrame), (
        "Dataset.to_pandas() returned an iterator (batched=True path) — "
        "this loader only handles a single in-memory DataFrame."
    )
    return (
        _load_data_split(df, split, one_hot=one_hot)
        if split
        else _load_data(df, one_hot=one_hot)
    )
