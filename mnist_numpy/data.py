from math import ceil
from pathlib import Path
from typing import Final, overload

import numpy as np
import pandas as pd

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEFAULT_DATA_PATH: Final[Path] = DATA_DIR / "mnist_train.csv"
MAX_PIXEL_VALUE: Final[int] = 255
N_DIGITS: Final[int] = 10
OUTPUT_PATH: Final[Path] = DATA_DIR / "output"
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RUN_PATH: Final[Path] = DATA_DIR / "run"
if not RUN_PATH.exists():
    RUN_PATH.mkdir(parents=True, exist_ok=True)


def _load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    return (
        np.eye(N_DIGITS)[df.iloc[:, 0].to_numpy()],
        np.array(df.iloc[:, 1:]) / MAX_PIXEL_VALUE,
    )


def _load_data_split(
    data_path: Path, split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split < 0 or split > 1:
        raise ValueError("Split must be between 0 and 1")
    df = pd.read_csv(data_path)
    split_index: int = ceil(len(df) * split)
    training_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    Y_train = np.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
    Y_test = np.eye(N_DIGITS)[test_set.iloc[:, 0].to_numpy()]

    X_train = np.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_test = np.array(test_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_test, Y_test


@overload
def load_data(data_path: Path, split: None = None) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def load_data(
    data_path: Path, split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def load_data(
    data_path: Path, split: float | None = None
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray]
):
    return (
        _load_data_split(data_path, split)
        if split is not None
        else _load_data(data_path)
    )
