import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return (exp_x := np.exp(x - np.max(x, axis=1, keepdims=True))) / np.sum(
        exp_x, axis=1, keepdims=True
    )


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(Y_true * np.log(Y_pred))


def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def deriv_ReLU(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)
