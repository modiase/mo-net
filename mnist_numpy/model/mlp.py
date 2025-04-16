from __future__ import annotations

import pickle
from collections.abc import Callable, MutableSequence
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from typing import IO, Literal, Protocol, Self, Sequence, cast

import numpy as np
from more_itertools import last

from mnist_numpy.functions import (
    ReLU,
    cross_entropy,
    get_activation_fn,
    identity,
    softmax,
)
from mnist_numpy.model import ModelBase
from mnist_numpy.model.layer import (
    DenseLayer,
    HiddenLayerBase,
    InputLayer,
    Layer,
    NonInputLayer,
    OutputLayerBase,
    RawOutputLayer,
    SoftmaxOutputLayer,
)
from mnist_numpy.types import (
    ActivationFn,
    Activations,
    LossContributor,
    PreActivations,
)


class GradientProto(Protocol):
    @property
    def _W(self) -> np.ndarray: ...

    @property
    def _B(self) -> np.ndarray: ...

    def __mul__(self, other: Self | float): ...

    def __rmul__(self, other: Self | float): ...

    def __add__(self, other: Self | float): ...

    def __radd__(self, other: Self | float): ...

    def __neg__(self): ...

    def __sub__(self, other: Self | float): ...

    def __truediv__(self, other: Self | float): ...

    def __pow__(self, other: float): ...


class MultiLayerPerceptron(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        @dataclass(frozen=True, kw_only=True)
        class Layer:
            activation_fn_name: str
            neurons: int
            parameters: tuple[np.ndarray, np.ndarray]

        layers: tuple[Layer, ...]

    @dataclass
    class Gradient:
        dParams: Sequence[GradientProto]  # TODO: improve-types

        def __mul__(self, other: Self | float):
            match other:
                case self.__class__():
                    return self.__class__(
                        dParams=tuple(other * param for param in self.dParams)
                    )
                case float():
                    return self.__class__(
                        dParams=tuple(other * param for param in self.dParams)
                    )
                case _:
                    return NotImplemented

        def __rmul__(self, other: Self | float):
            return self.__mul__(other)

        def __add__(self, other: Self | float):
            match other:
                case self.__class__():
                    return self.__class__(
                        dParams=tuple(
                            param1 + param2
                            for param1, param2 in zip(self.dParams, other.dParams)
                        )
                    )
                case float():
                    return self.__class__(
                        dParams=tuple(param + other for param in self.dParams)
                    )
                case _:
                    return NotImplemented

        def __radd__(self, other: Self):
            return self.__add__(other)

        def __neg__(self):
            return self.__class__(dParams=tuple(-param for param in self.dParams))

        def __sub__(self, other: Self):
            return self.__add__(-other)

        def __truediv__(self, other: Self | float):
            match other:
                case self.__class__():
                    return self.__class__(
                        dParams=tuple(
                            param1 / param2
                            for param1, param2 in zip(self.dParams, other.dParams)
                        )
                    )
                case float():
                    return self.__class__(
                        dParams=tuple(param / other for param in self.dParams)
                    )
                case _:
                    return NotImplemented

        def __pow__(self, other: float):
            return self.__class__(dParams=tuple(param**other for param in self.dParams))

    @classmethod
    def get_name(cls) -> str:
        return "mlp"

    @classmethod
    def get_description(cls) -> str:
        return "MultiLayer Perceptron"

    @classmethod
    def of(
        cls,
        *,
        layer_neuron_counts: Sequence[int],
        activation_fn: ActivationFn = identity,
        output_layer_type: Literal["softmax", "raw"] = "softmax",
        regularisers: Sequence[Callable[[Self], None]] = (),
    ) -> Self:
        if len(layer_neuron_counts) < 2:
            raise ValueError(f"{cls.__name__} must have at least 2 layers.")
        OutputLayerClass = (
            SoftmaxOutputLayer if output_layer_type == "softmax" else RawOutputLayer
        )
        layers: MutableSequence[Layer] = [InputLayer(neurons=layer_neuron_counts[0])]
        for layer_neuron_count in layer_neuron_counts[1:-1]:
            layers.append(
                DenseLayer(
                    neurons=layer_neuron_count,
                    activation_fn=activation_fn,
                    previous_layer=layers[-1],
                    parameters=DenseLayer.Parameters.he(
                        dim_in=layers[-1].neurons, dim_out=layer_neuron_count
                    )
                    if activation_fn == ReLU
                    else DenseLayer.Parameters.xavier(
                        dim_in=layers[-1].neurons, dim_out=layer_neuron_count
                    ),
                )
            )
        layers.append(
            OutputLayerClass(
                neurons=layer_neuron_counts[-1],
                previous_layer=layers[-1],
                parameters=DenseLayer.Parameters.xavier(
                    dim_in=layers[-1].neurons, dim_out=layer_neuron_counts[-1]
                ),
            )
        )
        model = cls(tuple(layers))
        for regulariser in regularisers:
            regulariser(model)
        return model

    def __init__(self, layers: Sequence[Layer]):  # noqa: F821
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
                f"Expected all layers except input and output layers to be {HiddenLayerBase.__name__}."
            )
        self._hidden_layers = cast(Sequence[HiddenLayerBase], layers[1:-1])
        self._As: MutableSequence[Activations] = []
        self._Zs: MutableSequence[PreActivations] = []
        self._input_layer = layers[0]
        self._output_layer = layers[-1]
        self._loss_contributors: MutableSequence[LossContributor] = []

    @property
    def loss_contributors(self) -> Sequence[LossContributor]:
        return self._loss_contributors

    def register_loss_contributor(self, contributor: LossContributor) -> None:
        self._loss_contributors.append(contributor)

    @property
    def non_input_layers(self) -> Sequence[NonInputLayer]:
        return tuple(chain(self._hidden_layers, (self._output_layer,)))

    @property
    def hidden_layers(self) -> Sequence[HiddenLayerBase]:
        return self._hidden_layers

    def accept_hidden_layer_visitor(
        self, visitor: Callable[[HiddenLayerBase, int], HiddenLayerBase]
    ) -> None:
        self._hidden_layers = tuple(
            visitor(layer, index) for index, layer in enumerate(self.hidden_layers)
        )

    @property
    def input_layer(self) -> InputLayer:
        return self._input_layer

    @property
    def output_layer(self) -> OutputLayerBase:
        return self._output_layer

    @property
    def layers(self) -> Sequence[Layer]:
        return tuple(
            chain((self.input_layer,), self.hidden_layers, (self.output_layer,))
        )

    def forward_prop(self, X: np.ndarray) -> Activations:
        self._Zs.clear()
        self._As.clear()

        Z, A = itemgetter(slice(-1), -1)(
            self.input_layer.forward_prop(As_prev=Activations(X))
        )
        self._Zs.append(
            last(Z)
        )  # If we ever need the pre-transformed pre-activations they are stored in Z
        self._As.append(A)

        for layer in self.non_input_layers:
            Z, A = itemgetter(slice(-1), -1)(layer.forward_prop(As_prev=last(self._As)))
            self._Zs.append(last(Z))
            self._As.append(A)

        return Activations(A)

    def backward_prop(self, Y_true: np.ndarray) -> MultiLayerPerceptron.Gradient:
        dps = []
        dp, dZ = self.output_layer.backward_prop(
            Y_pred=last(self._As),
            Y_true=Y_true,
            As_prev=last(self._As[:-1]),
            Zs_prev=last(self._Zs[:-1]),
        )
        dps.append(dp)
        for layer, As_prev, Zs_prev in zip(
            reversed(self.hidden_layers),
            reversed(self._As[:-2]),
            reversed(self._Zs[:-2]),
        ):
            dp, dZ = layer.backward_prop(As_prev=As_prev, Zs_prev=Zs_prev, dZ=dZ)
            dps.append(dp)
        return self.Gradient(dParams=tuple(reversed(dps)))  # type: ignore[arg-type] # TODO: Fix-types

    def update_parameters(self, update: MultiLayerPerceptron.Gradient) -> None:
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

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(
            self.Serialized(
                layers=tuple(
                    chain(
                        # TODO: Generalize
                        (
                            self.Serialized.Layer(
                                activation_fn_name=layer._activation_fn.name,  # TODO: Relies on private member
                                neurons=layer.neurons,
                                parameters=layer._parameters,
                            )
                            for layer in self.hidden_layers
                        ),
                        (
                            self.Serialized.Layer(
                                activation_fn_name=softmax.name,  # TODO: Generalize
                                neurons=self.output_layer.neurons,
                                parameters=self.output_layer._parameters,
                            ),
                        ),
                    ),
                ),
            ),
            io,
        )

    @classmethod
    def load(cls, source: IO[bytes]) -> Self:
        serialized = pickle.load(source)
        hidden_layers, output_layer = itemgetter(slice(-1), -1)(serialized.layers)

        layers: MutableSequence[Layer] = [InputLayer(neurons=784)]
        for layer in hidden_layers:
            layers.append(
                DenseLayer(
                    neurons=layer.neurons,
                    activation_fn=get_activation_fn(layer.activation_fn_name),
                    parameters=layer.parameters,
                    previous_layer=layers[-1],
                )
            )
        layers.append(
            SoftmaxOutputLayer(
                neurons=output_layer.neurons,
                parameters=output_layer.parameters,
                previous_layer=layers[-1],
            )
        )
        return cls(tuple(layers))

    @classmethod
    def initialize(cls, *dims: int) -> Self:
        return cls.of(layer_neuron_counts=dims, activation_fn=ReLU)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_prop(X).argmax(axis=1)

    def empty_gradient(self) -> MultiLayerPerceptron.Gradient:
        return self.Gradient(
            dParams=tuple(layer.empty_parameters() for layer in self.non_input_layers)
        )

    def compute_loss(self, X: np.ndarray, Y_true: np.ndarray) -> float:
        return sum(
            (loss_contributor() for loss_contributor in self.loss_contributors),
            start=1 / X.shape[0] * cross_entropy(self.forward_prop(X), Y_true),
        )
