from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Generic, NewType, Protocol, Self, TypedDict, TypeVar, cast

import jax.numpy as jnp
from typing_extensions import runtime_checkable

Activations = NewType("Activations", jnp.ndarray)

_X = TypeVar("_X", bound=jnp.ndarray | float)


ActivationFnName = NewType("ActivationFnName", str)


class ActivationFn(Protocol):
    def __call__(self, x: _X) -> _X: ...

    def deriv(self, x: _X) -> _X: ...

    @property
    def name(self) -> ActivationFnName: ...


_Quantity = TypeVar("_Quantity")


class D(Protocol, Generic[_Quantity]):
    def __add__(self, other: _Quantity) -> _Quantity: ...
    def __radd__(self, other: _Quantity) -> _Quantity: ...
    def __sub__(self, other: _Quantity) -> _Quantity: ...
    def __rsub__(self, other: _Quantity) -> _Quantity: ...
    def __mul__(self, other: _Quantity | float) -> _Quantity: ...
    def __rmul__(self, other: _Quantity | float) -> _Quantity: ...
    def __truediv__(self, other: _Quantity) -> _Quantity: ...
    def __rtruediv__(self, other: _Quantity) -> _Quantity: ...


class EventLike(Protocol):
    def clear(self) -> None: ...
    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def wait(self, timeout: float = ...) -> bool: ...


class HasBiases(Protocol):
    biases: jnp.ndarray


class HasWeights(Protocol):
    weights: jnp.ndarray


type LossContributor = Callable[[], float]


class TrainingStepHandler:
    def pre_forward(self, input_activations: Activations) -> Activations:
        return input_activations

    def post_forward(self, output_activations: Activations) -> Activations:
        return output_activations

    def pre_backward(self, dZ: D[Activations]) -> D[Activations]:
        return dZ

    def post_backward(self, dZ: D[Activations]) -> D[Activations]:
        return dZ


@runtime_checkable
class SupportsReinitialisation(Protocol):
    def reinitialise(self) -> None: ...


@runtime_checkable
class SupportsGradientOperations(Protocol):
    def __mul__(self, other: float | Self) -> Self: ...

    def __rmul__(self, other: float | Self) -> Self: ...

    def __add__(self, other: Self | float) -> Self: ...

    def __radd__(self, other: Self | float): ...

    def __neg__(self): ...

    def __sub__(self, other: Self | float): ...

    def __truediv__(self, other: Self | float): ...

    def __pow__(self, other: float): ...


@runtime_checkable
class SupportsUpdateParameters(Protocol):
    def update_parameters(self) -> None: ...


@runtime_checkable
class HasParameterCount(Protocol):
    @property
    def parameter_count(self) -> int: ...


type RawGradientType = Sequence[SupportsGradientOperations]
type UpdateGradientType = RawGradientType


ParamType = TypeVar("ParamType", bound=SupportsGradientOperations)
ParamType_co = TypeVar("ParamType_co", bound=SupportsGradientOperations, covariant=True)


class GradCache(TypedDict, Generic[ParamType]):  # noqa: F821
    dP: D[ParamType] | None


CacheType_co = TypeVar("CacheType_co", bound=GradCache, covariant=True)


@runtime_checkable
class GradLayer(Protocol, Generic[ParamType_co, CacheType_co]):
    @property
    def parameters(self) -> ParamType_co: ...

    @property
    def cache(self) -> CacheType_co: ...

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None: ...

    def empty_gradient(self) -> D[ParamType_co]: ...


@runtime_checkable
class SupportsForwardProp(Protocol):
    def forward_prop(self, *, input_activations: Activations) -> Activations: ...


_T_co = TypeVar("_T_co", covariant=True)


class SupportsDeserialize(Protocol, Generic[_T_co]):
    def deserialize(
        self,
        *,
        training: bool = False,
    ) -> _T_co: ...


@runtime_checkable
class SupportsSerialize(Protocol, Generic[_T_co]):
    def serialize(self) -> SupportsDeserialize[_T_co]: ...


_SupportsGradientOperationsT = TypeVar(
    "_SupportsGradientOperationsT", bound=SupportsGradientOperations
)


def d(value: _SupportsGradientOperationsT) -> D[_SupportsGradientOperationsT]:
    return cast(D[_SupportsGradientOperationsT], value)


def d_op(
    value: D[Activations], op: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """
    This is a helper function to apply JAX operations to the value of a D[Activations] object.
    The type-checker is unable to recognise that D[Activations] is a JAX array, so we need to cast it.
    """
    return op(cast(jnp.ndarray, value))


type Dimensions = Sequence[int]


class HasDimensions(Protocol):
    @property
    def input_dimensions(self) -> Dimensions: ...

    @property
    def output_dimensions(self) -> Dimensions: ...

    @staticmethod
    def get_dimensions(
        dimensioned: HasDimensions,
    ) -> tuple[Dimensions, Dimensions]:
        return dimensioned.input_dimensions, dimensioned.output_dimensions


class NormalisationType(StrEnum):
    LAYER = "layer"
    BATCH = "batch"
    NONE = "none"
