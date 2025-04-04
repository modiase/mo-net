from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np


class Gradient(Protocol):
    dWs: Sequence[np.ndarray]
    dbs: Sequence[np.ndarray]


class HasCosineDistance(Protocol):
    def cosine_distance(self, other: Self) -> float: ...
