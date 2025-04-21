from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

import numpy as np

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron

ConfigT = TypeVar("ConfigT")
AfterTrainingStepHandler: TypeAlias = Callable[
    [MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient], None
]


class OptimizerBase(ABC, Generic[ModelT, ConfigT]):
    def __init__(self, config: ConfigT):
        self._config = config
        self._after_training_step: Sequence[AfterTrainingStepHandler] = ()

    def training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        do_update: bool = True,
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]:
        gradient, update = self._training_step(
            model=model,
            X_train_batch=X_train_batch,
            Y_train_batch=Y_train_batch,
            do_update=do_update,
        )
        for after_training_step in self._after_training_step:
            after_training_step(gradient, update)
        return gradient, update

    def register_after_training_step_handler(
        self,
        after_training_step: AfterTrainingStepHandler,
    ) -> None:
        self._after_training_step = (
            *self._after_training_step,
            after_training_step,
        )

    @abstractmethod
    def _training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        do_update: bool,
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]: ...

    """
    Returns the raw gradient and the update that was applied to the model.
    """

    @abstractmethod
    def report(self) -> str: ...

    @property
    @abstractmethod
    def learning_rate(self) -> float: ...


@dataclass(frozen=True, kw_only=True)
class NoConfig:
    learning_rate: float


class NoOptimizer(OptimizerBase[ModelT, NoConfig]):
    Config = NoConfig

    def _training_step(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]:
        model.forward_prop(X=X_train_batch)
        gradient = model.backward_prop(Y_true=Y_train_batch)
        model.update_parameters(
            update=(update := -self._config.learning_rate * gradient)
        )
        return gradient, update

    def report(self) -> str:
        return ""

    @property
    def learning_rate(self) -> float:
        return self._config.learning_rate
