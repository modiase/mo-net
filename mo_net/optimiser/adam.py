import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import jax.numpy as jnp
from loguru import logger

from mo_net.constants import EPSILON
from mo_net.model.model import Model
from mo_net.optimiser.base import Base
from mo_net.optimiser.scheduler import Scheduler
from mo_net.protos import GradLayer

DEFAULT_BETA_1: Final[float] = 0.9
DEFAULT_BETA_2: Final[float] = 0.999
DEFAULT_MAX_GRAD_NORM: Final[float] = 1.0


def _grad_norm_squared(dP: Any) -> jnp.ndarray:
    """Sum of squares across all jnp.ndarray fields of a Parameters dataclass
    (or the raw array if dP is itself one)."""
    if isinstance(dP, jnp.ndarray):
        return jnp.sum(dP**2)
    total = jnp.asarray(0.0)
    if dataclasses.is_dataclass(dP):
        for field in dataclasses.fields(dP):
            value = getattr(dP, field.name)
            if isinstance(value, jnp.ndarray):
                total = total + jnp.sum(value**2)
    return total


@dataclass(frozen=True, kw_only=True)
class Config:
    beta_1: float = DEFAULT_BETA_1
    beta_2: float = DEFAULT_BETA_2
    epsilon: float = EPSILON
    scheduler: Scheduler
    # Global L2 gradient-norm clip; set to None to disable.
    max_grad_norm: float | None = DEFAULT_MAX_GRAD_NORM


type ConfigType = Config


class AdaM(Base[Config]):
    """https://arxiv.org/abs/1412.6980"""

    Config = Config

    def __init__(
        self,
        *,
        model: Model,
        config: ConfigType,
    ):
        super().__init__(config=config, model=model)

        self._scheduler = config.scheduler
        self._global_learning_rate = self._scheduler(0)

        for layer in self._model.grad_layers:
            layer.cache["first_moment"] = layer.empty_gradient()
            layer.cache["second_moment"] = layer.empty_gradient()

        self._snapshot_first_moment: Sequence[jnp.ndarray] | None = None
        self._snapshot_second_moment: Sequence[jnp.ndarray] | None = None

    def gradient_operation(self, layer: GradLayer) -> None:
        cache = layer.cache
        logger.trace(f"Computing gradient operation for layer {layer}.")

        cache["first_moment"] = (
            self._config.beta_1 * cache["first_moment"]
            + (1 - self._config.beta_1) * cache["dP"]
        )

        cache["second_moment"] = (
            self._config.beta_2 * cache["second_moment"]
            + (1 - self._config.beta_2) * cache["dP"] ** 2
        )

        cache["dP"] = (
            -self._global_learning_rate
            * (
                cache["first_moment"]
                / (1 - self._config.beta_1**self._iterations + self._config.epsilon)
            )
            / (
                (
                    cache["second_moment"]
                    / (1 - self._config.beta_2**self._iterations + self._config.epsilon)
                )
                ** 0.5
                + self._config.epsilon
            )
        )

    def compute_update(self) -> None:
        self._iterations += 1
        self._global_learning_rate = self._scheduler(self._iterations)
        output_layer = self._model.output_module.output_layer
        self._clip_gradients(output_layer)
        for layer in self._model.grad_layers:
            self.gradient_operation(layer)
        # Also process output layer if it has gradients (e.g., HierarchicalSoftmaxOutputLayer)
        if isinstance(output_layer, GradLayer):
            self.gradient_operation(output_layer)

    def _clip_gradients(self, output_layer: Any) -> None:
        """Rescale all layer dPs in-place if their joint L2 norm exceeds the
        configured cap. No-op when max_grad_norm is None."""
        if self._config.max_grad_norm is None:
            return
        total_norm_sq = jnp.asarray(0.0)
        for layer in self._model.grad_layers:
            total_norm_sq = total_norm_sq + _grad_norm_squared(layer.cache["dP"])
        if isinstance(output_layer, GradLayer):
            total_norm_sq = total_norm_sq + _grad_norm_squared(output_layer.cache["dP"])
        scale = float(
            jnp.minimum(
                1.0,
                self._config.max_grad_norm
                / (jnp.sqrt(total_norm_sq) + self._config.epsilon),
            )
        )
        if scale >= 1.0:
            return
        for layer in self._model.grad_layers:
            layer.cache["dP"] = layer.cache["dP"] * scale
        if isinstance(output_layer, GradLayer):
            output_layer.cache["dP"] = output_layer.cache["dP"] * scale

    @property
    def learning_rate(self) -> float:
        return self._global_learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._global_learning_rate:.10f}"

    def advance_to(self, iteration: int) -> None:
        super().advance_to(iteration)
        self._global_learning_rate = self._scheduler(iteration)

    def snapshot(self) -> None:
        super().snapshot()
        self._snapshot_first_moment = tuple(
            layer.cache["first_moment"] for layer in self._model.grad_layers
        )
        self._snapshot_second_moment = tuple(
            layer.cache["second_moment"] for layer in self._model.grad_layers
        )

    def restore(self) -> None:
        super().restore()
        if self._snapshot_first_moment is None or self._snapshot_second_moment is None:
            raise RuntimeError("No snapshot to restore from.")
        for layer, snapshot_first_moment, snapshot_second_moment in zip(
            self._model.grad_layers,
            self._snapshot_first_moment,
            self._snapshot_second_moment,
            strict=True,
        ):
            layer.cache["first_moment"] = snapshot_first_moment
            layer.cache["second_moment"] = snapshot_second_moment
