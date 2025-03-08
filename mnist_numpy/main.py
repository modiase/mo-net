import pickle
import time
from pathlib import Path
from typing import Final

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from more_itertools import sample
from tqdm import tqdm

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
MAX_PIXEL_VALUE: Final[int] = 256
N_DIGITS: Final[int] = 10
DEFAULT_NUM_ITERATIONS: Final[int] = 10000
DEFAULT_LEARNING_RATE: Final[float] = 0.001


def prepare_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_index: int = int(len(df) * 0.8)
    training_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    Y_train = np.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
    Y_test = np.eye(N_DIGITS)[test_set.iloc[:, 0].to_numpy()]

    X_train = np.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_test = np.array(test_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_test, Y_test


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(Y_true * np.log(Y_pred))


def forward_prop(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


def backward_prop(
    X: np.ndarray,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    m = X.shape[0]
    dZ = Y_pred - Y_true
    dW = (1 / m) * X.T @ dZ
    db = (1 / m) * np.sum(dZ)
    return dW, db


def train(
    X: np.ndarray,
    Y: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.random.random((X.shape[1], Y.shape[1]))
    b = np.random.random((1, Y.shape[1]))

    last_update_time = time.time()
    for i in tqdm(range(n_iterations)):
        Z = X @ W + b
        A = softmax(Z)
        dW, db = backward_prop(X, Y, A)
        if i % 1000 == 0 and (time.time() - last_update_time) > 30:
            tqdm.write(f"Iteration {i}: Loss = {cross_entropy(A, Y)}")
            last_update_time = time.time()
            pickle.dump((W, b), open(model_path, "wb"))
        W -= learning_rate * dW
        b -= learning_rate * db

    return W, b


@click.command()
@click.option(
    "-a",
    "--learning-rate",
    help="Set the learning rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
)
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    default=None,
)
@click.option(
    "-f", "--force-retrain", is_flag=True, help="Force retraining of the model"
)
@click.option(
    "-n",
    "--num-iterations",
    help="Set number of iterations",
    type=int,
    default=DEFAULT_NUM_ITERATIONS,
)
def main(
    *,
    model_path: Path | None,
    force_retrain: bool = False,
    num_iterations: int,
    learning_rate: float,
) -> None:
    X_train, Y_train, X_test, Y_test = prepare_data(
        pd.read_csv(DATA_DIR / "mnist_test.csv")
    )
    seed = int(time.time())
    np.random.seed(seed)

    if model_path is None:
        model_path = DATA_DIR / f"model_{seed=}_{num_iterations=}_{learning_rate=}.pkl"

    if not model_path.exists() or force_retrain:
        W, b = train(
            X_train,
            Y_train,
            learning_rate,
            num_iterations,
            model_path,
        )

        pickle.dump((W, b), open(model_path, "wb"))
    else:
        W, b = pickle.load(open(model_path, "rb"))
        W = W.reshape(28, 28, 10)
        avg = np.average(X_train, axis=0).reshape(28, 28)
        idx = list(range(10))

        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(np.multiply(W[:, :, i], avg), cmap="gray")
            plt.title(str(i))
            plt.axis("off")
        plt.show()

    Y_pred = np.argmax(softmax(forward_prop(X_train, W, b)), axis=1)
    Y_true = np.argmax(Y_train, axis=1)
    print(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = np.argmax(softmax(forward_prop(X_test, W, b)), axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    print(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    sample_indices = sample(range(len(X_test)), 10)
    for idx, i in enumerate(sample_indices):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {Y_pred[i]}, True: {Y_true[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
