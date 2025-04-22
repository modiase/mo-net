from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from typing import Generic, Protocol, Self, TypeVar, cast

import numpy as np
from more_itertools import last

from mnist_numpy.functions import ReLU, get_activation_fn, identity, softmax
from mnist_numpy.types import (
    ActivationFn,
    Activations,
    D,
    HasWeightsAndBiases,
    TrainingStepHandler,
    _ParamType,
)

_LayerType_co = TypeVar("_LayerType_co", bound="_LayerBase", covariant=True)


class SerializedLayer(Protocol, Generic[_LayerType_co]):
    @abstractmethod
    def deserialize(
        self,
        *,
        previous_layer: Layer,
        training: bool = False,
    ) -> _LayerType_co: ...


class _LayerBase(ABC, Generic[_ParamType]):
    Parameters: type[_ParamType]
    _parameters: _ParamType
    _previous_layer: Layer

    def __init__(
        self,
        *,
        neurons: int,
        activation_fn: ActivationFn,
    ):
        self._activation_fn = activation_fn
        self._neurons = neurons
        self._training_step_handlers: Sequence[TrainingStepHandler] = ()

    def register_training_step_handler(self, handler: TrainingStepHandler) -> None:
        self._training_step_handlers = tuple(
            chain(self._training_step_handlers, (handler,))
        )

    def forward_prop(self, *, As_prev: Activations) -> tuple[Activations, ...]:
        return tuple(
            reduce(
                lambda acc, handler: acc + [handler(last(acc))],  # type: ignore[operator]
                chain(
                    (handler.post_forward for handler in self._training_step_handlers),
                    (self._activation_fn,),
                ),
                [
                    self._forward_prop(
                        As_prev=(
                            reduce(
                                lambda acc, handler: handler(acc),
                                tuple(
                                    handler.pre_forward
                                    for handler in self._training_step_handlers
                                ),
                                As_prev,
                            )
                        )
                    )
                ],
            )
        )

    @abstractmethod
    def _forward_prop(self, *, As_prev: Activations) -> Activations: ...

    def _update_parameters(self, params: D[_ParamType]) -> None:
        self._parameters = params + self._parameters

    @property
    @abstractmethod
    def neurons(self) -> int: ...

    @abstractmethod
    def empty_parameters(self) -> _ParamType: ...

    @abstractmethod
    def reinitialise(self) -> None: ...

    @property
    def parameters(self) -> _ParamType:
        return self._parameters

    @abstractmethod
    def serialize(self) -> SerializedLayer[Self]: ...


@dataclass(kw_only=True, frozen=True)
class DenseParameters(HasWeightsAndBiases):
    _W: np.ndarray
    _B: np.ndarray

    def __getitem__(
        self, index: int | tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._W[index], self._B[index]

    def __add__(self, other: Self | float) -> Self:
        match other:
            case self.__class__():
                return self.__class__(_W=self._W + other._W, _B=self._B + other._B)
            case float():
                return self.__class__(_W=self._W + other, _B=self._B + other)
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float) -> Self:
        return self.__add__(other)

    def __neg__(self) -> Self:
        return self.__class__(_W=-self._W, _B=-self._B)

    def __sub__(self, other: Self | float) -> Self:
        return self.__add__(-other)

    def __rsub__(self, other: Self | float) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: float) -> Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__class__(_W=other * self._W, _B=other * self._B)

    def __rmul__(self, other: float) -> Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__mul__(other)

    def __truediv__(self, other: Self | float) -> Self:
        match other:
            case self.__class__():
                return self.__class__(
                    _W=self._W / other._W,
                    _B=self._B / other._B,
                )
            case float():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float) -> Self:
        return self.__class__(_W=self._W**scalar, _B=self._B**scalar)

    @classmethod
    def empty(cls, *, dim_in: int = 0, dim_out: int = 0) -> Self:
        return cls(_W=np.zeros((dim_in, dim_out)), _B=np.zeros(dim_out))

    @classmethod
    def random(cls, dim_in: int, dim_out: int) -> Self:
        return cls(_W=np.random.randn(dim_in, dim_out), _B=np.zeros(dim_out))

    @classmethod
    def xavier(cls, dim_in: int, dim_out: int) -> Self:
        return cls(
            _W=np.random.randn(dim_in, dim_out) * np.sqrt(1 / dim_in),
            _B=np.zeros(dim_out),
        )

    @classmethod
    def he(cls, dim_in: int, dim_out: int) -> Self:
        return cls(
            _W=np.random.normal(0, np.sqrt(2 / dim_in), (dim_in, dim_out)),
            _B=np.zeros(dim_out),
        )

    @classmethod
    def eye(cls, dim: int) -> Self:
        return cls(_W=np.eye(dim), _B=np.zeros(dim))

    @classmethod
    def of(cls, W: np.ndarray, B: np.ndarray) -> Self:
        return cls(_W=np.atleast_2d(W), _B=np.atleast_1d(B))


class HiddenLayerBase(
    _LayerBase[_ParamType]
):  # TODO: Consider merging this with OutputLayerBase
    def backward_prop(
        self, *, As_prev: Activations, Zs_prev: Activations, dZ: D[Activations]
    ) -> tuple[D[_ParamType], D[Activations]]:
        return reduce(
            lambda acc, handler: handler(*acc),
            reversed(
                tuple(handler.post_backward for handler in self._training_step_handlers)
            ),
            self._backward_prop(
                As_prev=As_prev,
                Zs_prev=Zs_prev,
                dZ=reduce(
                    lambda acc, handler: handler(acc),
                    reversed(
                        tuple(
                            handler.pre_backward
                            for handler in self._training_step_handlers
                        )
                    ),
                    dZ,
                ),
            ),
        )

    @abstractmethod
    def _backward_prop(
        self,
        *,
        As_prev: Activations,
        Zs_prev: Activations,
        dZ: D[Activations],
    ) -> tuple[D[_ParamType], D[Activations]]: ...

    # TODO: Implement ActivationLayer
    # The current implementaiton of backprop is clumsy because we did not
    # separate the application of the activation function from the dense layer.
    # Now that we are defining a convention where a layer can opt to 'participate'
    # in the backward pass by returning a tuple of (dP, dX), we can separate
    # the application of the activation function from the dense layer.
    #
    # We will define a new class, ActivationLayer, which will apply the activation
    # function to the output of the previous layer. It will not have any parameters.
    # It will not participate in the backward pass. We can then dispense with the
    # 'Zs_prev' argument in the backward_prop method.


class DenseLayer(HiddenLayerBase[DenseParameters]):
    Parameters = DenseParameters
    _parameters: DenseParameters

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        neurons: int
        activation_fn: str
        parameters: DenseParameters

        def deserialize(
            self,
            *,
            previous_layer: Layer,
            training: bool = False,
        ) -> DenseLayer:
            del training  # unused
            return DenseLayer(
                neurons=self.neurons,
                activation_fn=get_activation_fn(self.activation_fn),
                parameters=self.parameters,
                previous_layer=previous_layer,
            )

    def __init__(
        self,
        *,
        neurons: int,
        activation_fn: ActivationFn,
        parameters: DenseParameters | None = None,
        previous_layer: Layer,
    ):
        super().__init__(
            neurons=neurons,
            activation_fn=activation_fn,
        )
        self._previous_layer = previous_layer
        self._parameters = (
            parameters
            if parameters is not None
            else self.Parameters.random(previous_layer.neurons, self._neurons)
        )

    def _forward_prop(self, *, As_prev: Activations) -> Activations:
        return As_prev @ self._parameters._W + self._parameters._B

    def _backward_prop(
        self,
        *,
        As_prev: Activations,
        Zs_prev: Activations,
        dZ: D[Activations],
    ) -> tuple[D[DenseParameters], D[Activations]]:
        return cast(  # TODO: fix-types
            D[DenseParameters],
            self.Parameters.of(
                W=As_prev.T @ dZ,
                B=np.sum(
                    cast(  # TODO: fix-types
                        float | np.ndarray, dZ
                    ),
                    axis=0,
                ),
            ),
        ), dZ @ self._parameters._W.T * self._activation_fn.deriv(Zs_prev)

    @property
    def neurons(self) -> int:
        return self._neurons

    def empty_parameters(self) -> DenseParameters:
        return self.Parameters.empty(
            dim_in=self._previous_layer.neurons, dim_out=self._neurons
        )

    def reinitialise(self) -> None:
        self._parameters = (
            self.Parameters.he(
                dim_in=self._previous_layer.neurons, dim_out=self._neurons
            )
            if self._activation_fn == ReLU
            else self.Parameters.xavier(
                dim_in=self._previous_layer.neurons, dim_out=self._neurons
            )
        )

    def serialize(self) -> SerializedLayer[DenseLayer]:
        return self.Serialized(
            neurons=self._neurons,
            activation_fn=self._activation_fn.name,
            parameters=self._parameters,
        )


class OutputLayerBase(_LayerBase[_ParamType]):
    @abstractmethod
    def backward_prop(
        self,
        *,
        Y_pred: Activations,
        Y_true: np.ndarray,
        As_prev: Activations,
        Zs_prev: Activations,
    ) -> tuple[D[DenseParameters], D[Activations]]: ...

    @abstractmethod
    def _backward_prop(
        self,
        *,
        As_prev: Activations,
        Zs_prev: Activations,
        dZ: D[Activations],
    ) -> tuple[D[DenseParameters], D[Activations]]: ...

    # TODO: This can now be unified with DenseLayer
    @property
    def parameters(self) -> _ParamType:
        return self._parameters


class SoftmaxOutputLayer(OutputLayerBase[DenseParameters]):
    Parameters = DenseParameters

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        neurons: int
        activation_fn: str
        parameters: DenseParameters

        def deserialize(
            self,
            *,
            previous_layer: Layer,
            training: bool = False,
        ) -> SoftmaxOutputLayer:
            del training  # unused
            return SoftmaxOutputLayer(
                neurons=self.neurons,
                parameters=self.parameters,
                previous_layer=previous_layer,
            )

    def __init__(
        self,
        *,
        neurons: int,
        parameters: DenseParameters | None = None,
        previous_layer: Layer,
    ):
        super().__init__(
            neurons=neurons,
            activation_fn=softmax,
        )
        self._previous_layer = previous_layer
        self._parameters = (
            parameters
            if parameters is not None
            else self.Parameters.random(previous_layer.neurons, self._neurons)
        )

    def backward_prop(
        self,
        *,
        Y_pred: Activations,
        Y_true: np.ndarray,
        As_prev: Activations,
        Zs_prev: Activations,
    ) -> tuple[D[DenseParameters], D[Activations]]:
        return reduce(
            lambda acc, handler: handler(*acc),
            reversed(
                tuple(handler.post_backward for handler in self._training_step_handlers)
            ),
            self._backward_prop(
                As_prev=As_prev,
                Zs_prev=Zs_prev,
                dZ=reduce(
                    lambda acc, handler: handler(acc),
                    reversed(
                        tuple(
                            handler.pre_backward
                            for handler in self._training_step_handlers
                        )
                    ),
                    np.atleast_1d(Y_pred - Y_true),  # Specific to softmax
                ),
            ),
        )

    def _backward_prop(
        self,
        *,
        As_prev: Activations,
        Zs_prev: Activations,
        dZ: D[Activations],
    ) -> tuple[D[DenseParameters], D[Activations]]:
        # TODO: This can now be unified with DenseLayer
        return cast(  # TODO: fix-types
            D[DenseParameters],
            self.Parameters.of(
                W=As_prev.T @ dZ,
                B=np.sum(
                    cast(  # TODO: fix-types
                        float | np.ndarray, dZ
                    ),
                    axis=0,
                ),
            ),
        ), dZ @ self._parameters._W.T * self._activation_fn.deriv(Zs_prev)

    def _forward_prop(self, *, As_prev: Activations) -> Activations:
        return As_prev @ self._parameters._W + self._parameters._B

    @property
    def neurons(self) -> int:
        return self._neurons

    def empty_parameters(self) -> DenseParameters:
        return self.Parameters.empty(
            dim_in=self._previous_layer.neurons, dim_out=self._neurons
        )

    @property
    def parameters(self) -> DenseParameters:
        return self._parameters

    def reinitialise(self) -> None:
        self._parameters = self.Parameters.xavier(
            dim_in=self._previous_layer.neurons, dim_out=self._neurons
        )

    def serialize(self) -> SerializedLayer[SoftmaxOutputLayer]:
        return self.Serialized(
            neurons=self._neurons,
            activation_fn=self._activation_fn.name,
            parameters=self._parameters,
        )


class RawOutputLayer(SoftmaxOutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        neurons: int
        activation_fn: str
        parameters: DenseParameters

        def deserialize(
            self,
            *,
            previous_layer: Layer,
            training: bool = False,
        ) -> RawOutputLayer:
            del training  # unused
            return RawOutputLayer(
                neurons=self.neurons,
                parameters=self.parameters,
                previous_layer=previous_layer,
            )

    def __init__(
        self,
        *,
        neurons: int,
        previous_layer: Layer,
        parameters: DenseParameters | None = None,
    ):
        super().__init__(
            neurons=neurons,
            previous_layer=previous_layer,
            parameters=parameters,
        )
        self._activation_fn = identity


class InputLayer:
    def __init__(
        self,
        *,
        neurons: int,
    ):
        self._neurons = neurons
        self._training_step_handlers = ()
        self._activation_fn = identity

    def forward_prop(self, *, As_prev: Activations) -> tuple[Activations, ...]:
        return _LayerBase.forward_prop(self, As_prev=As_prev)  # type: ignore[arg-type]

    def _forward_prop(self, *, As_prev: Activations) -> Activations:
        return As_prev

    @property
    def neurons(self) -> int:
        return self._neurons


Layer = InputLayer | _LayerBase
NonInputLayer = HiddenLayerBase | OutputLayerBase
