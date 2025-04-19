from dataclasses import dataclass
from itertools import chain
from typing import ClassVar, Self

import numpy as np

from mnist_numpy.functions import identity
from mnist_numpy.model.layer import HiddenLayerBase
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import Activations, D


class BatchNormLayer(HiddenLayerBase):
    _EPSILON: ClassVar[float] = 1e-8

    @dataclass(frozen=True, kw_only=True)
    class Parameters:
        _gamma: np.ndarray
        _beta: np.ndarray

        def __pow__(self, other: float) -> Self:
            return self.__class__(
                _gamma=self._gamma**other,
                _beta=self._beta**other,
            )

        def __truediv__(self, other: Self | float) -> Self:
            if isinstance(other, float):
                return self.__class__(
                    _gamma=self._gamma / other,
                    _beta=self._beta / other,
                )
            if isinstance(other, self.__class__):
                return self.__class__(
                    _gamma=self._gamma / other._gamma,
                    _beta=self._beta / other._beta,
                )
            return NotImplemented

        def __add__(self, other: Self | float) -> Self:
            if isinstance(other, float):
                return self.__class__(
                    _gamma=self._gamma + other,
                    _beta=self._beta + other,
                )
            return self.__class__(
                _gamma=self._gamma + other._gamma,
                _beta=self._beta + other._beta,
            )

        def __radd__(self, other: Self | float) -> Self:
            return self.__add__(other)

        def __neg__(self) -> Self:
            return self.__class__(
                _gamma=-self._gamma,
                _beta=-self._beta,
            )

        def __sub__(self, other: Self | float) -> Self:
            if isinstance(other, float):
                return self.__class__(
                    _gamma=self._gamma - other,
                    _beta=self._beta - other,
                )
            return self.__class__(
                _gamma=self._gamma - other._gamma,
                _beta=self._beta - other._beta,
            )

        def __mul__(self, other: float) -> Self:
            return self.__class__(
                _gamma=self._gamma * other,
                _beta=self._beta * other,
            )

        def __rmul__(self, other: float) -> Self:
            return self.__mul__(other)

        @classmethod
        def empty(cls, *, neurons: int) -> Self:
            return cls(
                _gamma=np.ones(neurons),
                _beta=np.zeros(neurons),
            )

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _gamma: np.ndarray
        _beta: np.ndarray

    def __init__(
        self,
        *,
        neurons: int,
        momentum: float = 0.9,
        batch_size: int,
    ):
        super().__init__(neurons=neurons, activation_fn=identity)
        self._momentum = momentum
        self._running_mean = None
        self._running_variance = None
        self._cache = None
        self._batch_size = batch_size
        self._parameters = self.empty_parameters()
        self._neurons = neurons

    def _forward_prop(self, *, As_prev: Activations) -> Activations:
        batch_mean = np.mean(As_prev, axis=0)
        batch_variance = np.var(As_prev, axis=0)

        if self._running_mean is None:
            self._running_mean = batch_mean
            self._running_variance = batch_variance
        else:
            self._running_mean = (
                self._momentum * self._running_mean + (1 - self._momentum) * batch_mean
            )
            self._running_variance = (
                self._momentum * self._running_variance
                + (1 - self._momentum) * batch_variance
            )

        X_norm = (As_prev - batch_mean) / np.sqrt(batch_variance + self._EPSILON)

        self._cache = {
            "X": As_prev,
            "X_norm": X_norm,
            "mean": batch_mean,
            "var": batch_variance,
        }

        return self._parameters._gamma * X_norm + self._parameters._beta

    def _backward_prop(
        self,
        *,
        As_prev: Activations,
        Zs_prev: Activations,
        dZ: D[Activations],
    ) -> tuple[D[Parameters], D[Activations]]:
        del As_prev, Zs_prev  # unused
        X = self._cache["X"]
        mean = self._cache["mean"]
        var = self._cache["var"]

        dX_norm = dZ / self._parameters._gamma
        d_gamma = np.sum(dZ * self._cache["X_norm"], axis=0)
        d_beta = np.sum(dZ, axis=0)
        dP = self.Parameters(
            _gamma=d_gamma,
            _beta=d_beta,
        )

        dvar = 0.5 * np.sum(
            dX_norm * (X - mean) * np.power(var + self._EPSILON, -1.5), axis=0
        )
        dmean = (
            np.sum(dX_norm / np.sqrt(var + self._EPSILON), axis=0)
            + dvar * np.sum(-2 * (X - mean), axis=0) / self._batch_size
        )

        dX = (
            dX_norm / np.sqrt(var + self._EPSILON)
            + dvar * 2 * (X - mean) / self._batch_size
            + dmean / self._batch_size
        )

        return (dP, dX)

    def empty_gradient(self) -> Parameters:
        return self.Parameters(
            _gamma=np.ones_like(self._parameters._gamma),
            _beta=np.zeros_like(self._parameters._beta),
        )

    def empty_parameters(self) -> Parameters:
        return self.Parameters.empty(neurons=self._neurons)

    def update_parameters(self, update: Parameters) -> None:
        self._parameters += update

    def reinitialise(self) -> None:
        self._parameters = self.empty_parameters()

    @property
    def neurons(self) -> int:
        return self._neurons


class BatchNormRegulariser:
    def __init__(
        self,
        *,
        momentum: float = 0.9,
        batch_size: int,
    ):
        self._momentum = momentum
        self._batch_size = batch_size

    def __call__(self, model: MultiLayerPerceptron) -> None:
        model._hidden_layers = tuple(
            chain.from_iterable(
                (
                    # TODO: perhaps introduce the idea of a layer group
                    layer,
                    BatchNormLayer(
                        momentum=self._momentum,
                        batch_size=self._batch_size,
                        neurons=layer.neurons,
                    ),
                )
                for layer in model.hidden_layers
            )
        )
