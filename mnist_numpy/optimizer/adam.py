from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from mnist_numpy.model.model import Model
from mnist_numpy.optimizer.base import Base
from mnist_numpy.optimizer.scheduler import Scheduler
from mnist_numpy.protos import GradLayer

DEFAULT_BETA_1: Final[float] = 0.9
DEFAULT_BETA_2: Final[float] = 0.999
DEFAULT_EPSILON: Final[float] = 1e-8


@dataclass(frozen=True, kw_only=True)
class Config:
    beta_1: float = DEFAULT_BETA_1
    beta_2: float = DEFAULT_BETA_2
    epsilon: float = DEFAULT_EPSILON
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

        self._snapshot_first_moment: Sequence[np.ndarray] | None = None
        self._snapshot_second_moment: Sequence[np.ndarray] | None = None

    def gradient_operation(self, layer: GradLayer) -> None:
        cache = layer.cache
        cache["first_moment"] = (
            self._config.beta_1 * cache["first_moment"]
            + (1 - self._config.beta_1) * cache["dP"]
        )
        cache["second_moment"] = (
            self._config.beta_2 * cache["second_moment"]
            + (1 - self._config.beta_2) * cache["dP"] ** 2
        )
        cache["dP"] = -self.alpha_t * (
            cache["first_moment"]
            / (cache["second_moment"] ** 0.5 + self._config.epsilon)
        )

    def compute_update(self) -> None:
        self._iterations += 1
        self._global_learning_rate = self._scheduler(self._iterations)
        for layer in self._model.grad_layers:
            self.gradient_operation(layer)

    @property
    def alpha_t(self) -> float:
        return self._global_learning_rate * (
            (1 - self._config.beta_2**self._iterations) ** 0.5
            / (1 - self._config.beta_1**self._iterations)
        )

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
