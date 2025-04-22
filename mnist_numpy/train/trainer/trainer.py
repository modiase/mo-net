import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ContextManager, Final, TypeAlias, cast

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from mnist_numpy.config import TrainingParameters
from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.model.scheduler import CosineScheduler, WarmupScheduler
from mnist_numpy.optimizer import OptimizerBase, OptimizerConfigT
from mnist_numpy.optimizer.adam import AdamOptimizer
from mnist_numpy.optimizer.base import NoOptimizer
from mnist_numpy.train.batcher import Batcher
from mnist_numpy.train.context import (
    TrainingContext,
    set_training_progress,
    training_context,
)
from mnist_numpy.train.monitor import Monitor
from mnist_numpy.train.tracer import PerEpochTracerStrategy, Tracer, TracerConfig

DEFAULT_LOG_INTERVAL_SECONDS: Final[int] = 10


@dataclass(frozen=True, kw_only=True)
class TrainingResult:
    model_checkpoint_path: Path


def get_optimizer(
    optimizer_type: str,
    model: ModelT,
    training_parameters: TrainingParameters,
) -> OptimizerBase[ModelT, Any]:
    if optimizer_type == "adam":
        return AdamOptimizer(
            model=model,
            config=AdamOptimizer.Config(
                scheduler=WarmupScheduler.of(
                    training_parameters=training_parameters,
                    next_scheduler=CosineScheduler.of(
                        training_parameters=training_parameters,
                    ),
                ),
            ),
        )
    elif optimizer_type == "no":
        return NoOptimizer(
            config=NoOptimizer.Config(
                learning_rate=training_parameters.learning_rate_limits[1]
            ),
        )
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_type}")


AfterTrainingStepHandler: TypeAlias = Callable[
    [MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient], None
]


class BasicTrainer:
    def __init__(
        self,
        *,
        model: ModelT,
        optimizer: OptimizerBase[ModelT, OptimizerConfigT],
        training_parameters: TrainingParameters,
        training_log_path: Path,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._training_parameters = training_parameters
        self._training_log_path = training_log_path
        self._X_train = X_train
        self._Y_train = Y_train
        self._X_test = X_test
        self._Y_test = Y_test
        self._after_training_step: Sequence[AfterTrainingStepHandler] = ()

    def subscribe_to_after_training_step(
        self,
        subscription_handler: AfterTrainingStepHandler,
    ) -> None:
        self._after_training_step = (
            *self._after_training_step,
            subscription_handler,
        )

    def _create_training_loop_context(self) -> ContextManager[None]:
        return nullcontext()

    def train(self) -> TrainingResult:
        if not self._training_log_path.exists():
            self._training_log = pd.DataFrame(
                columns=[
                    "epoch",
                    "training_loss",
                    "monotonic_training_loss",
                    "test_loss",
                    "monotonic_test_loss",
                    "learning_rate",
                    "timestamp",
                ]
            )
            self._training_log.to_csv(self._training_log_path, index=False)
        else:
            self._training_log = pd.read_csv(self._training_log_path)

        logger.info(
            f"Training model {self._model.__class__.__name__}"
            f" for {self._training_parameters.num_epochs=} iterations"
            f" using optimizer {self._optimizer.__class__.__name__}."
        )

        self._model_checkpoint_path = self._training_log_path.with_name(
            self._training_log_path.name.replace("training_log.csv", "partial.pkl")
        )
        self._model_training_parameters_path = self._training_log_path.with_name(
            self._training_log_path.name.replace(
                "training_log.csv", "training_parameters.json"
            )
        )
        if not self._model_training_parameters_path.exists():
            self._model_training_parameters_path.write_text(
                self._training_parameters.model_dump_json()
            )
        else:
            self._training_parameters = TrainingParameters.model_validate_json(
                self._model_training_parameters_path.read_text()
            )

        logger.info(
            f"Training model..\nSaving partial results to: {self._model_checkpoint_path}."
        )
        logger.info(f"\n{self._training_parameters=}.")
        logger.info(f"\n{self._training_log_path=}.")
        self._model.dump(open(self._model_checkpoint_path, "wb"))

        self._L_train_min = self._model.compute_loss(
            X=self._X_train, Y_true=self._Y_train
        )
        self._L_test_min = self._model.compute_loss(X=self._X_test, Y_true=self._Y_test)

        if self._training_parameters.trace_logging:
            tracer = Tracer(
                model=cast(MultiLayerPerceptron, self._model),  # TODO: Fix-types
                training_log_path=self._training_log_path,
                tracer_config=TracerConfig(
                    trace_strategy=PerEpochTracerStrategy(
                        training_set_size=self._training_parameters.train_set_size,
                        batch_size=self._training_parameters.batch_size,
                    ),
                ),
            )
            self.subscribe_to_after_training_step(tracer)

        self._monitor = Monitor(
            X_train=self._X_train,
            Y_train=self._Y_train,
            history_max_len=self._training_parameters.history_max_len,
            warmup_batches=self._training_parameters.warmup_epochs
            * self._training_parameters.batches_per_epoch,
        )
        self.subscribe_to_after_training_step(self._monitor.post_batch)

        self._before_training_loop()

        training_context.set(
            TrainingContext(
                training_progress=0.0, model_checkpoint_path=self._model_checkpoint_path
            )
        )
        with self._create_training_loop_context():
            self._training_loop()

        return TrainingResult(
            model_checkpoint_path=self._model_checkpoint_path,
        )

    def _before_training_loop(self) -> None:
        self._batcher = Batcher(
            X=self._X_train,
            Y=self._Y_train,
            batch_size=self._training_parameters.batch_size,
        )

    def _training_loop(self) -> None:
        last_log_time = time.time()
        for i in tqdm(
            range(
                self._training_parameters.total_batches,
            ),
            total=self._training_parameters.total_batches,
        ):
            with set_training_progress(self._training_parameters.current_progress(i)):
                gradient, update = self._training_step()
                for handler in self._after_training_step:
                    handler(gradient, update)

                if i % self._training_parameters.batches_per_epoch == 0:
                    L_train = self._model.compute_loss(
                        X=self._X_train, Y_true=self._Y_train
                    )
                    L_test = self._model.compute_loss(
                        X=self._X_test, Y_true=self._Y_test
                    )

                    self._L_train_min = min(self._L_train_min, L_train)
                    if L_test < self._L_test_min:
                        self._model.dump(open(self._model_checkpoint_path, "wb"))
                        self._L_test_min = L_test
                    self._monitor.post_epoch(L_test)

                    pd.DataFrame(
                        [
                            [
                                self._training_parameters.current_epoch(i),
                                L_train,
                                self._L_train_min,
                                L_test,
                                self._L_test_min,
                                self._optimizer.learning_rate,
                                datetime.now(),
                            ]
                        ],
                        columns=self._training_log.columns,
                    ).to_csv(
                        self._training_log_path, mode="a", header=False, index=False
                    )

                if time.time() - last_log_time > DEFAULT_LOG_INTERVAL_SECONDS:
                    tqdm.write(
                        f"Iteration {i}, Epoch {self._training_parameters.current_epoch(i)}, Training Loss = {L_train}, Test Loss = {L_test}"
                        + (
                            f", {report}"
                            if (report := self._optimizer.report()) != ""
                            else ""
                        )
                    )
                    last_log_time = time.time()

    def _training_step(
        self,
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]:
        X_train_batch, Y_train_batch = next(self._batcher)
        gradient = self._optimizer.training_step(
            model=self._model,
            X_train_batch=X_train_batch,
            Y_train_batch=Y_train_batch,
        )
        update = self._optimizer.compute_update(gradient)
        self._model.update_parameters(update)
        return gradient, update
