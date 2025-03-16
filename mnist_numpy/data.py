from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
N_DIGITS: Final[int] = 10
MAX_PIXEL_VALUE: Final[int] = 255
DEFAULT_DATA_PATH: Final[Path] = DATA_DIR / "mnist_test.csv"


def load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    split_index: int = int(len(df) * 0.8)
    training_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    Y_train = np.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
    Y_test = np.eye(N_DIGITS)[test_set.iloc[:, 0].to_numpy()]

    X_train = np.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_test = np.array(test_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_test, Y_test
