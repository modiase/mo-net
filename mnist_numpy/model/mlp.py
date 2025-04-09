from __future__ import annotations

import pickle
from collections.abc import MutableSequence
from dataclasses import dataclass
from itertools import chain
from typing import IO, ClassVar, Literal, Protocol, Self, Sequence, cast

import numpy as np
from more_itertools import first, last, pairwise, triplewise

from mnist_numpy.functions import ReLU, eye, softmax
from mnist_numpy.model import ModelBase
from mnist_numpy.model.layer import (
    DenseLayer,
    HiddenLayerBase,
    InputLayer,
    LayerBase,
    OutputLayerBase,
    RawOutputLayer,
    SoftmaxOutputLayer,
)
from mnist_numpy.types import ActivationFn, Activations, PreActivations


@dataclass(frozen=True, kw_only=True)
class MLP_Gradient:
    dWs: Sequence[np.ndarray]
    dbs: Sequence[np.ndarray]

    def __add__(self, other: Self) -> Self:
        return self.__class__(
            dWs=tuple(dW1 + dW2 for dW1, dW2 in zip(self.dWs, other.dWs)),
            dbs=tuple(db1 + db2 for db1, db2 in zip(self.dbs, other.dbs)),
        )

    def __neg__(self) -> Self:
        return self.__class__(
            dWs=tuple(-dW for dW in self.dWs),
            dbs=tuple(-db for db in self.dbs),
        )

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __mul__(self, other: float) -> Self:
        return self.__class__(
            dWs=tuple(dW * other for dW in self.dWs),
            dbs=tuple(db * other for db in self.dbs),
        )

    def __rmul__(self, other: float) -> Self:
        return self * other

    def __truediv__(self, other: float) -> Self:
        return self * (1 / other)

    def __pow__(self, exp: float) -> Self:
        return self.__class__(
            dWs=tuple(dW**exp for dW in self.dWs),
            dbs=tuple(db**exp for db in self.dbs),
        )

    def __getitem__(self, idx: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        match idx:
            case int():
                return self.dWs[idx], self.dbs[idx]
            case tuple():
                i, *rest = idx
                return self.dWs[i][*rest], self.dbs[i][*rest]

    def cosine_distance(self, other: Self) -> float:
        v1 = np.concat(tuple(chain((w.flatten() for w in self.dWs), self.dbs)))
        v2 = np.concat(tuple(chain((w.flatten() for w in other.dWs), other.dbs)))
        return (1 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))) / 2


@dataclass(frozen=True, kw_only=True)
class MLP_Parameters:
    W: Sequence[np.ndarray]
    b: Sequence[np.ndarray]

    def __add__(self, other: Self | MLP_Gradient) -> Self:
        match other:
            case MLP_Parameters():
                return self.__class__(
                    W=tuple(W1 + W2 for W1, W2 in zip(self.W, other.W)),
                    b=tuple(b1 + b2 for b1, b2 in zip(self.b, other.b)),
                )
            case MLP_Gradient():
                return self.__class__(
                    W=tuple(W1 + dW for W1, dW in zip(self.W, other.dWs)),
                    b=tuple(b1 + db for b1, db in zip(self.b, other.dbs)),
                )
            case _:
                raise ValueError(f"Invalid type: {type(other)}")

    def unroll(self) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        return tuple(w.flatten() for w in self.W), tuple(b for b in self.b)

    def __getitem__(self, idx: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        match idx:
            case int():
                return self.W[idx], self.b[idx]
            case tuple():
                i, *rest = idx
                return self.W[i][*rest], self.b[i][*rest]


class MultilayerPerceptron(ModelBase[MLP_Parameters, MLP_Gradient]):
    Gradient = MLP_Gradient

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
            [np.zeros(dim_out) for dim_out in dims[1:]],
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
        return tuple(Z), tuple(chain(map(ReLU, Z[:-1]), (Z[-1],)))

    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: Sequence[np.ndarray],
        A: Sequence[np.ndarray],
    ) -> MLP_Gradient:
        Y_pred = softmax(Z[-1])
        dZ = Y_pred - Y_true
        _A = [X, *A]
        k = 1 / X.shape[0]

        dW = []
        db = []
        for idx in range(len(self._W) - 1, -1, -1):
            dW.append(k * (_A[idx].T @ dZ))
            db.append(k * np.sum(dZ, axis=0))
            if np.isnan(dW[-1]).any() or np.isnan(db[-1]).any() or np.isnan(dZ).any():
                raise ValueError("Invalid gradient. Aborting training.")
            if idx > 0:
                dZ = (dZ @ self._W[idx].T) * ReLU.deriv(Z[idx - 1])

        return MLP_Gradient(
            dWs=tuple(reversed(dW)),
            dbs=tuple(reversed(db)),
        )

    def update_parameters(
        self,
        update: MLP_Gradient,
    ) -> None:
        for w, dW in zip(self._W, update.dWs):
            w += dW
        for b, db in zip(self._b, update.dbs):
            b += db

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

    def empty_gradient(self) -> MLP_Gradient:
        return MLP_Gradient(
            dWs=tuple(np.zeros_like(w) for w in self._W),
            dbs=tuple(np.zeros_like(b) for b in self._b),
        )

    @property
    def parameters(self) -> MLP_Parameters:
        return MLP_Parameters(W=tuple(self._W), b=tuple(self._b))


class SupportsMultiplyByFloat(Protocol):
    def __mul__(self, scalar: float): ...

    def __rmul__(self, scalar: float): ...


class MultiLayerPerceptronV2:
    @dataclass
    class Gradient:
        dParams: Sequence[SupportsMultiplyByFloat]  # TODO: improve-types

        def __mul__(self, scalar: float):
            return self.__class__(
                dParams=tuple(scalar * param for param in self.dParams)
            )

        def __rmul__(self, scalar: float):
            return self.__mul__(scalar)

    @classmethod
    def of(
        cls,
        *,
        layer_neuron_counts: Sequence[int],
        activation_fn: ActivationFn = eye,
        output_layer_type: Literal["softmax", "raw"] = "softmax",
    ) -> Self:
        if len(layer_neuron_counts) < 2:
            raise ValueError(f"{cls.__name__} must have at least 2 layers.")
        OutputLayerClass = (
            SoftmaxOutputLayer if output_layer_type == "softmax" else RawOutputLayer
        )
        return cls(
            tuple(
                (
                    InputLayer(layer_neuron_counts[0]),
                    *(
                        DenseLayer(neurons=layer, activation_fn=activation_fn)
                        for layer in layer_neuron_counts[1:-1]
                    ),
                    OutputLayerClass(layer_neuron_counts[-1]),
                )
            )
        )

    def __init__(self, layers: Sequence[LayerBase]):
        if len(layers) < 2:
            raise ValueError(f"{self.__class__.__name__} must have at least 2 layers.")
        if not isinstance(layers[-1], OutputLayerBase):
            raise ValueError(
                f"{self.__class__.__name__} must have an output layer of type {OutputLayerBase.__name__}."
            )
        if not isinstance(layers[0], InputLayer):
            raise ValueError(
                f"{self.__class__.__name__} must have an input layer of type {InputLayer.__name__}."
            )
        if not all(isinstance(layer, HiddenLayerBase) for layer in layers[1:-1]):
            raise ValueError(
                f"Hidden layers must have type {HiddenLayerBase.__name__}."
            )
        self._hidden_layers = cast(Sequence[HiddenLayerBase], layers[1:-1])
        self._As: MutableSequence[Activations] = []
        self._Zs: MutableSequence[PreActivations] = []
        self._input_layer = layers[0]
        self._output_layer = layers[-1]

        self._input_layer._init(
            previous_layer=None,
            next_layer=first(chain(self._hidden_layers, (self._output_layer,))),  # type: ignore[arg-type]
        )
        for previous_layer, hidden_layer, next_layer in triplewise(layers):
            hidden_layer._init(previous_layer=previous_layer, next_layer=next_layer)
        self._output_layer._init(
            previous_layer=last(chain((self._input_layer,), self._hidden_layers)),  # type: ignore[arg-type]
            next_layer=None,
        )

    @property
    def hidden_layers(self) -> Sequence[HiddenLayerBase]:
        return self._hidden_layers

    @property
    def non_input_layers(self) -> Sequence[HiddenLayerBase | OutputLayerBase]:
        return tuple(chain(self._hidden_layers, (self._output_layer,)))

    @property
    def input_layer(self) -> InputLayer:
        return self._input_layer

    @property
    def output_layer(self) -> OutputLayerBase:
        return self._output_layer

    @property
    def layers(self) -> Sequence[LayerBase]:
        return tuple(
            chain((self.input_layer,), self.hidden_layers, (self.output_layer,))
        )

    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        self._Zs.clear()
        self._As.clear()

        Z, A = self.input_layer._forward_prop(As=Activations(X))
        self._Zs.append(Z)
        self._As.append(A)

        for layer in self.non_input_layers:
            Z, A = layer._forward_prop(As=self._As[-1])
            self._Zs.append(Z)
            self._As.append(A)
        return A

    def backward_prop(self, Y_true: np.ndarray) -> MultiLayerPerceptronV2.Gradient:
        dps = []
        dp, dZ = self.output_layer._backward_prop(
            Y_pred=self._As[-1],
            Y_true=Y_true,
            As_prev=self._As[-2],
            Zs_prev=self._Zs[-2],
        )
        dps.append(dp)
        for layer, As_prev, Zs_prev in zip(
            reversed(self.hidden_layers),
            reversed(self._As[:-2]),
            reversed(self._Zs[:-2]),
        ):
            dp, dZ = layer._backward_prop(As_prev=As_prev, Zs_prev=Zs_prev, dZ=dZ)
            dps.append(dp)
        return self.Gradient(dParams=tuple(reversed(dps)))  # type: ignore[arg-type] # TODO: Fix-types

    def update_params(self, update: MultiLayerPerceptronV2.Gradient) -> None:
        def _it():
            return zip(update.dParams, self.non_input_layers)

        if not all(isinstance(upd, layer.__class__.Parameters) for upd, layer in _it()):
            raise ValueError(
                "Incompatible update passed to model."
                f" Update has types {', '.join(type(upd).__name__ for upd in update.dParams)}"
                f" Model has layers {', '.join(type(layer).__name__ for layer in self.non_input_layers)}"
            )
        for dP, layer in _it():
            layer._update_parameters(dP)
