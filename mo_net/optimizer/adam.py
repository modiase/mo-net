from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import jax.numpy as jnp
from loguru import logger

from mo_net.constants import EPSILON
from mo_net.model.model import Model
from mo_net.optimizer.base import Base
from mo_net.optimizer.scheduler import Scheduler
from mo_net.protos import GradLayer

DEFAULT_BETA_1: Final[float] = 0.9
DEFAULT_BETA_2: Final[float] = 0.999


@dataclass(frozen=True, kw_only=True)
class Config:
    beta_1: float = DEFAULT_BETA_1
    beta_2: float = DEFAULT_BETA_2
    epsilon: float = EPSILON
    scheduler: Scheduler


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
        for layer in self._model.grad_layers:
            self.gradient_operation(layer)

    @property
    def learning_rate(self) -> float:
        return self._global_learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._global_learning_rate:.10f}"

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
        ):
            layer.cache["first_moment"] = snapshot_first_moment
            layer.cache["second_moment"] = snapshot_second_moment
