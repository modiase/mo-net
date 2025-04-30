import random as rn
from itertools import product
from math import pi
from typing import Final

import numpy as np
import pandas as pd

from mnist_numpy.augment import rotate, shear, translate
from mnist_numpy.data import DEFAULT_DATA_PATH, load_data

MNIST_Y_SIZE: Final[int] = 28
MNIST_X_SIZE: Final[int] = 28
MNIST_SIZE: Final[int] = MNIST_Y_SIZE * MNIST_X_SIZE

if __name__ == "__main__":
    X_train, Y_train, _, _ = load_data(DEFAULT_DATA_PATH)
    columns = [
        "label",
        *(
            f"{x}x{y}"
            for x, y in product(range(1, MNIST_X_SIZE + 1), range(1, MNIST_Y_SIZE + 1))
        ),
    ]
    data = []
    for i in range(100000):
        idx = rn.randint(0, 7999)
        digit = translate(
            shear(
                rotate(
                    a=X_train[idx].reshape(MNIST_Y_SIZE, MNIST_X_SIZE),
                    theta=rn.random() * pi / 2 - pi / 4,
                    x_size=MNIST_X_SIZE,
                    y_size=MNIST_Y_SIZE,
                ),
                x_shear=rn.random() / 2,
                y_shear=rn.random() / 2,
                x_size=MNIST_X_SIZE,
                y_size=MNIST_Y_SIZE,
            ),
            x_offset=rn.randint(-3, 3),
            y_offset=rn.randint(-3, 3),
            x_size=MNIST_X_SIZE,
            y_size=MNIST_Y_SIZE,
        )
        label = np.argmax(Y_train[idx])
        data.append([label, *digit.reshape(MNIST_SIZE)])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv("mnist_hard.csv", index=False)
