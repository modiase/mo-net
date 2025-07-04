from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from loguru import logger

from mo_net.constants import EPSILON
from mo_net.model.model import Model
from mo_net.optimizer.base import Base
from mo_net.optimizer.scheduler import Scheduler
from mo_net.protos import GradLayer

DEFAULT_BETA: Final[float] = 0.9


@dataclass(frozen=True, kw_only=True)
class Config:
    beta: float = DEFAULT_BETA
    epsilon: float = EPSILON
    scheduler: Scheduler


type ConfigType = Config


class RMSProp(Base[Config]):
    """RMSProp optimizer for adaptive learning rates.

    RMSProp adapts the learning rate by using a moving average of squared gradients
    to normalize the gradient. This helps with training stability, especially for
    non-convex optimization problems.
    """

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
            layer.cache["squared_grad_avg"] = layer.empty_gradient()

        self._snapshot_squared_grad_avg: Sequence[np.ndarray] | None = None

    def gradient_operation(self, layer: GradLayer) -> None:
        cache = layer.cache
        logger.trace(f"Computing gradient operation for layer {layer}.")

        cache["squared_grad_avg"] = (
            self._config.beta * cache["squared_grad_avg"]
            + (1 - self._config.beta) * cache["dP"] ** 2
        )

        cache["dP"] = (
            -self._global_learning_rate
            * cache["dP"]
            / (cache["squared_grad_avg"] ** 0.5 + self._config.epsilon)
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
        self._snapshot_squared_grad_avg = tuple(
            layer.cache["squared_grad_avg"] for layer in self._model.grad_layers
        )

    def restore(self) -> None:
        super().restore()
        if self._snapshot_squared_grad_avg is None:
            raise RuntimeError("No snapshot to restore from.")
        for layer, snapshot_squared_grad_avg in zip(
            self._model.grad_layers,
            self._snapshot_squared_grad_avg,
        ):
            layer.cache["squared_grad_avg"] = snapshot_squared_grad_avg
