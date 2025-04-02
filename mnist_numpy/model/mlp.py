import pickle
from dataclasses import dataclass
from typing import IO, ClassVar, Self, Sequence

import numpy as np
from more_itertools import pairwise

from mnist_numpy.functions import ReLU, deriv_ReLU, softmax
from mnist_numpy.model import ModelBase

MLP_WeightsT = tuple[Sequence[np.ndarray], Sequence[np.ndarray]]
MLP_GradientT = tuple[Sequence[np.ndarray], Sequence[np.ndarray]]


class MultilayerPerceptron(ModelBase[MLP_WeightsT, MLP_GradientT]):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _tag: ClassVar[str] = "mlp_relu"
        W: tuple[np.ndarray, ...]
        b: tuple[np.ndarray, ...]

    @property
    def layers(self) -> Sequence[int]:
        return tuple(w.shape[0] for w in self._W)

    def get_name(self) -> str:
        return f"mlp_relu_{'_'.join(str(layer) for layer in self.layers[1:])}"

    @classmethod
    def get_description(cls) -> str:
        return "Multilayer Perceptron with ReLU activation"

    @classmethod
    def initialize(cls, *dims: int) -> Self:
        return cls(
            [np.random.randn(dim_in, dim_out) for dim_in, dim_out in pairwise(dims)],
            [np.random.randn(dim_out) for dim_out in dims[1:]],
        )

    def __init__(
        self,
        W: Sequence[np.ndarray],
        b: Sequence[np.ndarray],
    ):
        self._W = list(W)
        self._b = list(b)

    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        Z = [X @ self._W[0] + self._b[0]]
        for w, b in zip(self._W[1:], self._b[1:]):
            Z.append(ReLU(Z[-1]) @ w + b)
        return tuple(Z), tuple(map(ReLU, Z))

    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: Sequence[np.ndarray],
        A: Sequence[np.ndarray],
    ) -> MLP_GradientT:
        Y_pred = softmax(Z[-1])
        dZ = Y_pred - Y_true
        _A = [X, *A]
        k = 1 / X.shape[0]

        dW = []
        db = []
        for idx in range(len(self._W) - 1, -1, -1):
            dW.append(k * (_A[idx].T @ dZ))
            db.append(k * np.sum(dZ, axis=0))
            if (
                np.isnan(dW[-1]).any()
                or np.isnan(db[-1]).any()
                or np.isnan(dZ[-1]).any()
            ):
                raise ValueError("Invalid gradient. Aborting training.")
            if idx > 0:
                dZ = (dZ @ self._W[idx].T) * deriv_ReLU(Z[idx - 1])

        return tuple(reversed(dW)), tuple(reversed(db))

    def update_weights(
        self,
        weights: MLP_WeightsT,
    ) -> None:
        dWs, dbs = weights
        for w, dW in zip(self._W, dWs):
            w -= dW
        for b, db in zip(self._b, dbs):
            b -= db

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(self.Serialized(W=tuple(self._W), b=tuple(self._b)), io)

    @classmethod
    def load(cls, source: IO[bytes] | Serialized) -> Self:
        if isinstance(source, cls.Serialized):
            return cls(W=source.W, b=source.b)
        data = pickle.load(source)
        if data._tag != cls.Serialized._tag:
            raise ValueError(f"Invalid model type: {data._tag}")
        return cls(W=data.W, b=data.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(softmax(self._forward_prop(X)[1][-1]), axis=1)

    def empty_weights(self) -> MLP_WeightsT:
        return (
            tuple(np.zeros_like(w) for w in self._W),
            tuple(np.zeros_like(b) for b in self._b),
        )
