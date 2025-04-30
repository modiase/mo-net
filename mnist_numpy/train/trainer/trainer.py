import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ContextManager, Final, cast

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from mnist_numpy.config import TrainingParameters
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.optimizer import Base, OptimizerConfigT
from mnist_numpy.optimizer.adam import AdaM
from mnist_numpy.optimizer.base import Null
from mnist_numpy.optimizer.scheduler import CosineScheduler, WarmupScheduler
from mnist_numpy.train.batcher import Batcher
from mnist_numpy.train.context import (
    TrainingContext,
    set_model_checkpoint_save_epoch,
    set_training_progress,
    training_context,
)
from mnist_numpy.train.monitor import Monitor
from mnist_numpy.train.tracer import PerEpochTracerStrategy, Tracer, TracerConfig
from mnist_numpy.types import SupportsGradientOperations, UpdateGradientType

DEFAULT_LOG_INTERVAL_SECONDS: Final[int] = 10


@dataclass(frozen=True, kw_only=True)
class TrainingResult:
    model_checkpoint_path: Path


def get_optimizer(
    optimizer_type: str,
    model: MultiLayerPerceptron,
    training_parameters: TrainingParameters,
) -> Base[Any]:
    if optimizer_type == "adam":
        return AdaM(
            model=model,
            config=AdaM.Config(
                scheduler=WarmupScheduler.of(
                    training_parameters=training_parameters,
                    next_scheduler=CosineScheduler.of(
                        training_parameters=training_parameters,
                    ),
                ),
            ),
        )
    elif optimizer_type == "no":
        return Null(
            model=model,
            config=Null.Config(
                learning_rate=training_parameters.learning_rate_limits[1]
            ),
        )
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_type}")


type AfterTrainingStepHandler = Callable[
    [Sequence[SupportsGradientOperations], Sequence[SupportsGradientOperations]],
    None,
]


class BasicTrainer:
    def __init__(
        self,
        *,
        disable_shutdown: bool = False,
        model: MultiLayerPerceptron,
        optimizer: Base[OptimizerConfigT],
        training_parameters: TrainingParameters,
        start_epoch: int | None = None,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> None:
        self._disable_shutdown = disable_shutdown
        self._model = model
        self._start_epoch = start_epoch if start_epoch is not None else 0
        self._optimizer = optimizer
        self._training_parameters = training_parameters
        self._X_train = X_train
        self._Y_train = Y_train
        self._X_test = X_test
        self._Y_test = Y_test
        self._after_training_step: Sequence[AfterTrainingStepHandler] = ()
        self._last_update: UpdateGradientType | None = None

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

    def _create_training_step_context(self) -> ContextManager[None]:
        if self._training_parameters.monotonic:
            return self._monotonic_training_step_context()
        return nullcontext()

    @contextmanager
    def _monotonic_training_step_context(self) -> Iterator[None]:
        loss_before = self._model.compute_loss(X=self._X_train, Y_true=self._Y_train)
        yield
        loss_after = self._model.compute_loss(X=self._X_train, Y_true=self._Y_train)
        if loss_after > loss_before:
            self._revert_training_step()

    def _revert_training_step(self) -> None:
        if self._last_update is None:
            raise ValueError("No update to revert.")
        self._model.populate_caches([-update for update in self._last_update])
        self._model.update_parameters()
        self._last_update = None

    def resume(
        self,
        start_epoch: int,
        model_checkpoint_path: Path,
    ) -> TrainingResult:
        logger.info(f"Resuming training from epoch {start_epoch}.")

        self._start_epoch = start_epoch
        self._model = MultiLayerPerceptron.load(
            open(model_checkpoint_path, "rb"), training=True
        )
        self._monitor.reset(restore_history=True)
        self._optimizer.set_model(self._model)
        self._optimizer.restore()
        with self._create_training_loop_context():
            self._training_loop()

        return TrainingResult(
            model_checkpoint_path=model_checkpoint_path,
        )

    def train(self) -> TrainingResult:
        if not self._training_parameters.log_path.exists():
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
            self._training_log.to_csv(self._training_parameters.log_path, index=False)
        else:
            self._training_log = pd.read_csv(self._training_parameters.log_path)

        logger.info(
            f"Training model {self._model.__class__.__name__}"
            f" for {self._training_parameters.num_epochs=} iterations"
            f" using optimizer {self._optimizer.__class__.__name__}."
        )
        logger.info(
            f"Model has dimensions: {self._model.dimensions} and parameter count: {self._model.parameter_count}."
        )

        self._model_checkpoint_path = self._training_parameters.log_path.with_name(
            self._training_parameters.log_path.name.replace(
                "training_log.csv", "partial.pkl"
            )
        )
        self._model_training_parameters_path = (
            self._training_parameters.log_path.with_name(
                self._training_parameters.log_path.name.replace(
                    "training_log.csv", "training_parameters.json"
                )
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

        logger.info(f"Saving partial results to: {self._model_checkpoint_path}.")
        logger.info(f"Training parameters: {self._training_parameters}.")
        logger.info(f"Training log path: {self._training_parameters.log_path}.")
        self._model.dump(open(self._model_checkpoint_path, "wb"))

        self._L_train_min = self._model.compute_loss(
            X=self._X_train, Y_true=self._Y_train
        )
        self._L_test_min = self._model.compute_loss(X=self._X_test, Y_true=self._Y_test)

        if self._training_parameters.trace_logging:
            tracer = Tracer(
                model=cast(MultiLayerPerceptron, self._model),  # TODO: Fix-types
                training_log_path=self._training_parameters.log_path,
                tracer_config=TracerConfig(
                    trace_strategy=PerEpochTracerStrategy(
                        training_set_size=self._training_parameters.train_set_size,
                        batch_size=self._training_parameters.batch_size,
                    ),
                ),
            )
            self.subscribe_to_after_training_step(tracer.post_batch)

        if not self._training_parameters.no_monitoring:
            self._monitor = Monitor(
                X_train=self._X_train,
                Y_train=self._Y_train,
                batches_per_epoch=self._training_parameters.batches_per_epoch,
                history_max_len=self._training_parameters.history_max_len,
                warmup_epochs=self._training_parameters.warmup_epochs,
            )
            self.subscribe_to_after_training_step(self._monitor.post_batch)

        self._before_training_loop()

        training_context.set(
            TrainingContext(
                training_progress=0.0,
                model_checkpoint_path=self._model_checkpoint_path,
                model_checkpoint_save_epoch=self._start_epoch,
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
                (
                    start_batch := self._start_epoch
                    * self._training_parameters.batches_per_epoch
                ),
                self._training_parameters.total_batches,
            ),
            initial=start_batch,
            total=self._training_parameters.num_epochs
            * self._training_parameters.batches_per_epoch,
            unit=" epoch",
            unit_scale=1 / self._training_parameters.batches_per_epoch,
            bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ):
            set_training_progress(self._training_parameters.current_progress(i))
            with self._create_training_step_context():
                gradient, update = self._training_step()
            for handler in self._after_training_step:
                handler(gradient, update)

            if i % self._training_parameters.batches_per_epoch == 0:
                L_train = self._model.compute_loss(
                    X=self._X_train, Y_true=self._Y_train
                )
                L_test = self._model.compute_loss(X=self._X_test, Y_true=self._Y_test)

                self._L_train_min = min(self._L_train_min, L_train)
                if L_test < self._L_test_min:
                    self._model.dump(open(self._model_checkpoint_path, "wb"))
                    self._monitor.snapshot()
                    self._optimizer.snapshot()
                    set_model_checkpoint_save_epoch(
                        self._training_parameters.current_epoch(i)
                    )
                    self._L_test_min = L_test
                self._post_epoch(L_test)
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
                    self._training_parameters.log_path,
                    mode="a",
                    header=False,
                    index=False,
                )

            if time.time() - last_log_time > DEFAULT_LOG_INTERVAL_SECONDS:
                tqdm.write(
                    f"Epoch {self._training_parameters.current_epoch(i)}, Training Loss = {L_train}, Test Loss = {L_test}"
                    + (
                        f", {report}"
                        if (report := self._optimizer.report()) != ""
                        else ""
                    )
                )
                last_log_time = time.time()

    def _training_step(
        self,
    ) -> tuple[
        Sequence[SupportsGradientOperations],
        Sequence[SupportsGradientOperations],
    ]:
        X_train_batch, Y_train_batch = next(self._batcher)
        gradient, update = self._optimizer.training_step(
            X_train_batch=X_train_batch,
            Y_train_batch=Y_train_batch,
            return_gradients=True,
        )
        if self._training_parameters.monotonic:
            self._last_update = update
        return gradient, update

    def _post_epoch(self, L_test: float) -> None:
        if not self._training_parameters.no_monitoring:
            self._monitor.post_epoch(L_test)

    def shutdown(self) -> None:
        pass
