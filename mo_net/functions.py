from collections.abc import Mapping
from typing import Any, Callable, Literal, Protocol, TypeVar

import jax
import jax.numpy as jnp

ArrayT = TypeVar("ArrayT", bound=jnp.ndarray)

type TransformFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

type LossFn = Callable[[jnp.ndarray, jnp.ndarray], float]


class ActivationFn(Protocol[ArrayT]):
    def __call__(self, x: ArrayT) -> ArrayT: ...

    @staticmethod
    def deriv(x: ArrayT) -> ArrayT: ...


def cross_entropy(Y_pred: jnp.ndarray, Y_true: jnp.ndarray) -> float:
    return -jnp.mean(
        jnp.sum(Y_true * jnp.log(jnp.clip(Y_pred, 1e-15, 1 - 1e-15)), axis=-1)
    ).item()


def sparse_cross_entropy(Y_pred: jnp.ndarray, Y_true: jnp.ndarray) -> float:
    return -jnp.mean(
        jnp.log(jnp.clip(Y_pred[jnp.arange(Y_true.shape[0]), Y_true], 1e-15, 1 - 1e-15))
    ).item()


def get_loss_fn(name: Literal["cross_entropy", "sparse_cross_entropy"]) -> LossFn:
    if name == "cross_entropy":
        return cross_entropy
    elif name == "sparse_cross_entropy":
        return sparse_cross_entropy
    else:
        raise ValueError(f"Unknown loss function: {name}")


def _identity(x: jnp.ndarray) -> jnp.ndarray:
    return x


class ReLU:
    def __call__(self, x: ArrayT) -> ArrayT:
        return jax.nn.relu(x)  # type: ignore[return-value]

    @staticmethod
    def deriv(x: ArrayT) -> ArrayT:
        return jnp.where(x > 0, 1, 0)  # type: ignore[return-value]


class Tanh:
    def __call__(self, x: ArrayT) -> ArrayT:
        return jax.nn.tanh(x)  # type: ignore[return-value]

    @staticmethod
    def deriv(x: ArrayT) -> ArrayT:
        return 1 - jax.nn.tanh(x) ** 2  # type: ignore[return-value]


class LeakyReLU:
    def __call__(self, x: ArrayT) -> ArrayT:
        return jax.nn.leaky_relu(x)  # type: ignore[return-value]

    @staticmethod
    def deriv(x: ArrayT) -> ArrayT:
        return jnp.where(x > 0, 1, 0.01)  # type: ignore[return-value]


class Identity:
    def __call__(self, x: ArrayT) -> ArrayT:
        return x  # type: ignore[return-value]

    @staticmethod
    def deriv(x: ArrayT) -> ArrayT:
        return jnp.ones_like(x)  # type: ignore[return-value]


class Softmax:
    def __call__(self, x: ArrayT) -> ArrayT:
        return jax.nn.softmax(x)  # type: ignore[return-value]

    @staticmethod
    def deriv(x: ArrayT) -> ArrayT:
        # Softmax derivative is more complex, but for backprop we typically
        # use the output directly, so this is a placeholder
        # In practice, softmax is usually handled in the loss function.
        return jnp.ones_like(x)  # type: ignore[return-value]


identity = Identity()
ACTIVATION_FUNCTIONS: Mapping[str, ActivationFn[jnp.ndarray]] = {
    "relu": ReLU(),
    "tanh": Tanh(),
    "softmax": Softmax(),
    "leaky_relu": LeakyReLU(),
    "identity": identity,
}


def get_activation_fn(name: str) -> ActivationFn[jnp.ndarray]:
    """Get activation function by name."""
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]


def parse_activation_fn(
    ctx: Any, param: Any, value: object
) -> ActivationFn[jnp.ndarray]:
    """Parse activation function from command line."""
    del ctx, param  # unused
    if value is None:
        raise ValueError("Activation function name is required")
    if not isinstance(value, str):
        raise ValueError("Activation function name must be a string")
    try:
        return get_activation_fn(value)
    except ValueError as e:
        raise ValueError(str(e)) from e
