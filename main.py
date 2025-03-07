from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR: Final[Path] = Path(__file__).parent / "data"



if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "mnist_test.csv")
    Y = df.iloc[:, 0]

    training_set = df.iloc[:8000, :]
    test_set = df.iloc[8000:, :]

    X_train = training_set.iloc[:, 1:]
    Y_train = training_set.iloc[:, 0]
    X_test = test_set.iloc[:, 1:]
    Y_test = test_set.iloc[:, 0]

    X_example = X_train.iloc[0].values.reshape(28, 28)
    plt.imshow(X_example, cmap='binary')
    plt.axis('off')
    plt.show()
    




