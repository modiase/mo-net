from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp

from mo_net.protos import ActivationFnName

type TransformFn = Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]

type LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def cross_entropy(Y_pred: jnp.ndarray, Y_true: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(
        jnp.sum(Y_true * jnp.log(jnp.clip(Y_pred, 1e-15, 1 - 1e-15)), axis=-1)
    )


def sparse_cross_entropy(Y_pred: jnp.ndarray, Y_true: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(
        jnp.log(jnp.clip(Y_pred[jnp.arange(Y_true.shape[0]), Y_true], 1e-15, 1 - 1e-15))
    )


def get_loss_fn(name: Literal["cross_entropy", "sparse_cross_entropy"]) -> LossFn:
    if name == "cross_entropy":
        return cross_entropy
    elif name == "sparse_cross_entropy":
        return sparse_cross_entropy
    else:
        raise ValueError(f"Unknown loss function: {name}")


def identity(x: jnp.ndarray) -> jnp.ndarray:
    return x


# Dictionary mapping activation names to JAX functions directly
ACTIVATION_FUNCTIONS = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "softmax": jax.nn.softmax,
    "leaky_relu": jax.nn.leaky_relu,
    "identity": identity,
}


def get_activation_fn(name: ActivationFnName) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get activation function by name."""
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]


def parse_activation_fn(
    ctx: Any, param: Any, value: object
) -> Callable[[jnp.ndarray], jnp.ndarray]:
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
