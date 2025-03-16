from itertools import product
import random as rn
from math import pi

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mnist_numpy.data import load_data


def rotate(a: np.ndarray, theta: float):
    y, x = np.mgrid[0:28, 0:28]

    x_centered = x - 14
    y_centered = y - 14

    x_source = x_centered * np.cos(-theta) - y_centered * np.sin(-theta) + 14
    y_source = x_centered * np.sin(-theta) + y_centered * np.cos(-theta) + 14

    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)

    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < 28)
        & (y_source_int >= 0)
        & (y_source_int < 28)
    )

    output = np.zeros((28, 28))

    output[y[valid_indices], x[valid_indices]] = a[
        y_source_int[valid_indices], x_source_int[valid_indices]
    ]

    return output


def shear(a: np.ndarray, x_shear: float, y_shear: float):
    y, x = np.mgrid[0:28, 0:28]

    x_centered = x - 14
    y_centered = y - 14

    x_source = x_centered + y_centered * x_shear + 14
    y_source = x_centered * y_shear + y_centered + 14

    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)

    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < 28)
        & (y_source_int >= 0)
        & (y_source_int < 28)
    )

    output = np.zeros((28, 28))

    output[y[valid_indices], x[valid_indices]] = a[
        y_source_int[valid_indices], x_source_int[valid_indices]
    ]

    return output


def translate(a: np.ndarray, x_trans: float, y_trans: float):
    y, x = np.mgrid[0:28, 0:28]

    x_source = x + x_trans
    y_source = y + y_trans

    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)

    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < 28)
        & (y_source_int >= 0)
        & (y_source_int < 28)
    )

    output = np.zeros((28, 28))

    output[y[valid_indices], x[valid_indices]] = a[
        y_source_int[valid_indices], x_source_int[valid_indices]
    ]

    return output


if __name__ == "__main__":
    X_train, Y_train, _, _ = load_data()
    columns = ["label", *(f"{x}x{y}" for x, y in product(range(1, 29), range(1, 29)))]
    data = []
    for i in range(10000):
        idx = rn.randint(0, 7999)
        digit = translate(
            shear(
                rotate(X_train[idx].reshape(28, 28), rn.random() * pi / 2 - pi / 4),
                rn.random() / 2,
                rn.random() / 2,
            ),
            rn.randint(-3, 3),
            rn.randint(-3, 3),
        )
        label = np.argmax(Y_train[idx])
        data.append([label, *digit.reshape(784)])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv("mnist_hard.csv", index=False)
