import pickle
import re
import time
from pathlib import Path
from typing import Final

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from more_itertools import sample
from tqdm import tqdm

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEFAULT_NUM_ITERATIONS: Final[int] = 10000
DEFAULT_LEARNING_RATE: Final[float] = 0.001
MAX_PIXEL_VALUE: Final[int] = 256
N_DIGITS: Final[int] = 10


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "mnist_test.csv")
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
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dZ = Y_pred - Y_true
    dW = X.T @ dZ
    db = np.sum(dZ)
    return dW, db


def train_model(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    model_path: Path,
    W: np.ndarray | None = None,
    b: np.ndarray | None = None,
    start_iteration: int = 0,
    training_log_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if W is None:
        W = np.random.random((X_train.shape[1], Y_train.shape[1]))
    if b is None:
        b = np.random.random((1, Y_train.shape[1]))

    last_update_time = time.time()
    if training_log_path is None:
        training_log_path = model_path.with_name(f"{model_path.stem}_training_log.csv")
        training_log = pd.DataFrame(columns=["iteration", "training_loss", "test_loss"])
        training_log.to_csv(training_log_path, index=False)
    else:
        training_log = pd.read_csv(training_log_path)

    for i in tqdm(range(start_iteration, n_iterations)):
        Z_train = X_train @ W + b
        L_train = (1 / X_train.shape[0]) * cross_entropy(softmax(Z_train), Y_train)
        dW, db = backward_prop(X_train, softmax(Z_train), Y_train)
        if i % 1000 == 0 and (time.time() - last_update_time) > 30:
            Z_test = X_test @ W + b
            L_test = (1 / X_test.shape[0]) * cross_entropy(softmax(Z_test), Y_test)
            tqdm.write(
                f"Iteration {i}: Training Loss = {L_train}, Test Loss = {L_test}"
            )
            pd.DataFrame([[i, L_train, L_test]], columns=training_log.columns).to_csv(
                training_log_path, mode="a", header=False, index=False
            )
            pickle.dump(
                (W, b),
                open(
                    model_path.with_name(f"{model_path.stem}_part.pkl"),
                    "wb",
                ),
            )
            last_update_time = time.time()

        W -= learning_rate * dW
        b -= learning_rate * db

    return W, b


@click.group()
def cli(): ...


@cli.command(help="Train the model")
@click.option(
    "-a",
    "--learning-rate",
    help="Set the learning rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
)
@click.option(
    "-t",
    "--training-log-path",
    type=Path,
    help="Set the path to the training log file",
    default=None,
)
@click.option(
    "-n",
    "--num-iterations",
    help="Set number of iterations",
    type=int,
    default=DEFAULT_NUM_ITERATIONS,
)
def train(
    *,
    training_log_path: Path | None,
    num_iterations: int,
    learning_rate: float,
) -> None:
    X_train, Y_train, X_test, Y_test = load_data()

    seed = int(time.time())
    np.random.seed(seed)

    if training_log_path is None:
        model_path = DATA_DIR / f"model_{seed=}_{num_iterations=}_{learning_rate=}.pkl"
    else:
        model_path = training_log_path.with_name(
            f"{training_log_path.stem.replace('_training_log', '')}.pkl"
        )

    W, b = train_model(
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        learning_rate=learning_rate,
        n_iterations=num_iterations
        if training_log_path is None
        else int(re.search(r"num_iterations=(\d+)", str(training_log_path)).group(1)),
        model_path=model_path,
        **(
            dict(
                zip(
                    ("W", "b"),
                    pickle.load(
                        open(
                            model_path.with_name(
                                f"{training_log_path.stem.replace('_training_log', '_part')}.pkl"
                            ),
                            "rb",
                        )
                    ),
                )
            )
            if training_log_path is not None
            else {"W": None, "b": None}
        ),
        training_log_path=training_log_path,
        start_iteration=0
        if training_log_path is None
        else pd.read_csv(training_log_path).iloc[-1, 0],
    )

    pickle.dump((W, b), open(model_path, "wb"))

    model_path.with_name(
        f"{training_log_path.stem.replace('_training_log', '_part')}.pkl"
    ).unlink(missing_ok=True)


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    required=True,
)
def infer(model_path: Path):
    X_train, Y_train, X_test, Y_test = load_data()
    W, b = pickle.load(open(model_path, "rb"))

    Y_pred = np.argmax(softmax(forward_prop(X_train, W, b)), axis=1)
    Y_true = np.argmax(Y_train, axis=1)
    print(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = np.argmax(softmax(forward_prop(X_test, W, b)), axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    print(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    sample_indices = sample(range(len(X_test)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {Y_pred[i]}, True: {Y_true[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


@cli.command(help="Run explainability analysis")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    required=True,
    type=Path,
)
def explain(*, model_path: Path):
    X_train = load_data()[0]
    W, b = pickle.load(open(model_path, "rb"))
    W = W.reshape(28, 28, 10)
    avg = np.average(X_train, axis=0).reshape(28, 28)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.multiply(W[:, :, i], avg), cmap="gray")
        plt.title(str(i))
        plt.axis("off")
    plt.show()


@cli.command(help="Sample input data", name="sample")
def sample_():
    X_train = load_data()[0]
    sample_indices = sample(range(len(X_train)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    cli()
