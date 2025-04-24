import warnings
from dataclasses import dataclass, field
from typing import Final

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.optimizer.base import Base
from mnist_numpy.optimizer.scheduler import NoopScheduler, Scheduler
from mnist_numpy.types import GradLayer

DEFAULT_BETA_1: Final[float] = 0.9
DEFAULT_BETA_2: Final[float] = 0.999
DEFAULT_EPSILON: Final[float] = 1e-8


@dataclass(frozen=True, kw_only=True)
class Config:
    beta_1: float = DEFAULT_BETA_1
    beta_2: float = DEFAULT_BETA_2
    epsilon: float = DEFAULT_EPSILON
    scheduler: Scheduler = field(default_factory=NoopScheduler)
    start_learning_rate: float = 0.0


type ConfigType = Config


class AdaM(Base[Config]):
    Config = Config

    def __init__(
        self,
        *,
        model: MultiLayerPerceptron,
        config: ConfigType,
    ):
        super().__init__(config=config, model=model)

        self._current_learning_rate: float = config.start_learning_rate
        self._scheduler = config.scheduler
        if (
            isinstance(self._scheduler, NoopScheduler)
            and self._current_learning_rate == 0.0
        ):
            warnings.warn("NoopScheduler is used, but start_learning_rate is 0.0. ")

        for layer in self._model.grad_layers:
            layer.cache["first_moment"] = layer.empty_gradient()
            layer.cache["second_moment"] = layer.empty_gradient()

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
        self._current_learning_rate = self._scheduler(
            self._iterations, self._current_learning_rate
        )
        for layer in self._model.grad_layers:
            self.gradient_operation(layer)

    @property
    def alpha_t(self) -> float:
        return self._current_learning_rate * (
            (1 - self._config.beta_2**self._iterations) ** 0.5
            / (1 - self._config.beta_1**self._iterations)
        )

    @property
    def learning_rate(self) -> float:
        return self._current_learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._current_learning_rate:.10f}"
