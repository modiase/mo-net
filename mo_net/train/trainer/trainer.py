import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Final, Literal, assert_never

import jax
import jax.numpy as jnp
from loguru import logger
from tqdm import tqdm

from mo_net.config import TrainingParameters
from mo_net.data import RUN_PATH
from mo_net.functions import LossFn, TransformFn
from mo_net.model.model import Model
from mo_net.optimizer import Base, OptimizerConfigT
from mo_net.optimizer.adam import AdaM
from mo_net.optimizer.base import Null
from mo_net.optimizer.rmsprop import RMSProp
from mo_net.optimizer.scheduler import CosineScheduler, WarmupScheduler
from mo_net.protos import SupportsGradientOperations, UpdateGradientType
from mo_net.train.batcher import IndexBatcher
from mo_net.train.exceptions import CheckFailed
from mo_net.train.monitor import Monitor
from mo_net.train.run import TrainingRun
from mo_net.train.tracer import PerEpochTracerStrategy, Tracer, TracerConfig

DEFAULT_LOG_INTERVAL_SECONDS: Final[int] = 10


@dataclass(frozen=True, kw_only=True)
class TrainingSuccessful:
    model_checkpoint_path: Path


@dataclass(frozen=True, kw_only=True)
class TrainingFailed:
    message: str
    model_checkpoint_path: Path
    model_checkpoint_save_epoch: int | None = None
    training_progress: float | None = None


type TrainingResult = TrainingSuccessful | TrainingFailed


type OptimizerType = Literal["adam", "none"]


def get_optimizer(
    optimizer_type: OptimizerType,
    model: Model,
    training_parameters: TrainingParameters,
) -> Base[Any]:
    match optimizer_type:
        case "adam":
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
        case "none":
            return Null(
                model=model,
                config=Null.Config(
                    learning_rate=training_parameters.learning_rate_limits[1]
                ),
            )
        case "rmsprop":
            return RMSProp(
                model=model,
                config=RMSProp.Config(
                    scheduler=WarmupScheduler.of(
                        training_parameters=training_parameters,
                        next_scheduler=CosineScheduler.of(
                            training_parameters=training_parameters,
                        ),
                    ),
                ),
            )
        case never:
            assert_never(never)


type AfterTrainingStepHandler = Callable[
    [Sequence[SupportsGradientOperations], Sequence[SupportsGradientOperations]],
    None | CheckFailed,
]


class BasicTrainer:
    def __init__(
        self,
        *,
        disable_shutdown: bool = False,
        model: Model,
        optimizer: Base[OptimizerConfigT],
        run: TrainingRun,
        training_parameters: TrainingParameters,
        transform_fn: TransformFn | None = None,
        start_epoch: int | None = None,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        X_val: jnp.ndarray,
        Y_val: jnp.ndarray,
        loss_fn: LossFn,
        key: jnp.ndarray,
    ) -> None:
        key1, key2 = jax.random.split(key, 2)
        self._disable_shutdown = disable_shutdown
        self._run = run
        self._model = model
        self._monitor: Monitor | None = None
        self._start_epoch = start_epoch if start_epoch is not None else 0
        self._optimizer = optimizer
        self._training_parameters = training_parameters
        self._X_train = X_train
        self._Y_train = Y_train
        self._transform = transform_fn
        self._loss_fn = loss_fn
        self._logger = logger.bind(name="trainer")
        self._batcher = IndexBatcher(
            train_set_size=X_train.shape[0],
            batch_size=self._training_parameters.batch_size,
            key=key1,
        )
        self._key = key2
        self._X_val = X_val
        self._Y_val = Y_val
        self._after_training_step: Sequence[AfterTrainingStepHandler] = ()
        self._last_update: UpdateGradientType | None = None
        self._L_val_min_epoch: int | None = None
        self._on_shutdown_handlers: Sequence[Callable[[], None]] = (
            lambda: self._run.end_run(),
        )

    def subscribe_to_after_training_step(
        self,
        subscription_handler: AfterTrainingStepHandler,
    ) -> None:
        self._after_training_step = (
            *self._after_training_step,
            subscription_handler,
        )

    def subscribe_to_shutdown(self, handler: Callable[[], None]) -> None:
        self._on_shutdown_handlers = (*self._on_shutdown_handlers, handler)

    def _create_training_loop_context(self) -> ContextManager[None]:
        @contextmanager
        def _training_loop_context() -> Iterator[None]:
            try:
                yield
            finally:
                for handler in self._on_shutdown_handlers:
                    try:
                        handler()
                    except Exception:
                        self._logger.exception("Error in shutdown handler.")
                if not self._disable_shutdown:
                    try:
                        self.shutdown()
                    except Exception:
                        self._logger.exception("Error in shutdown handler.")

        return _training_loop_context()

    def _create_training_step_context(self) -> ContextManager[None]:
        if self._training_parameters.monotonic:
            return self._monotonic_training_step_context()
        return nullcontext()

    @contextmanager
    def _monotonic_training_step_context(self) -> Iterator[None]:
        raise NotImplementedError("Monotonic training is not implemented.")

    def _revert_training_step(self) -> None:
        if self._last_update is None:
            raise ValueError("No update to revert.")
        self._model.populate_caches([-update for update in self._last_update])
        self._model.update_parameters()
        self._last_update = None

    def resume(
        self,
        *,
        model_checkpoint_path: Path,
        start_epoch: int,
    ) -> TrainingResult:
        self._logger.info(f"Resuming training from epoch {start_epoch}.")

        self._start_epoch = start_epoch
        self._model = Model.load(model_checkpoint_path, training=True)
        if self._monitor is not None:
            self._monitor.reset(restore_history=True)
        self._optimizer.set_model(self._model)
        self._optimizer.restore()

        with self._create_training_loop_context():
            return self._training_loop()

    def train(self) -> TrainingResult:
        self._logger.info(
            f"Training model {self._model.__class__.__name__}"
            f" for {self._training_parameters.num_epochs=} iterations"
            f" using optimizer {self._optimizer.__class__.__name__}."
        )
        self._logger.info(f"{self._model.print()}")
        self._logger.info(
            f"Model has dimensions: {', '.join(f'[{dim}]' for dim in self._model.module_dimensions)} and parameter count: {self._model.parameter_count}."
        )

        self._run.start_run(
            total_batches=self._training_parameters.total_batches,
            total_epochs=self._training_parameters.num_epochs,
        )
        self._model_checkpoint_path = Path(
            str((RUN_PATH / self._run.id).with_suffix(".pkl")).replace(
                "_model_training_log", ""
            )
        )
        self._run.log_training_parameters(
            training_parameters=self._training_parameters.model_dump_json()
        )

        self._logger.info(f"Saving partial results to: {self._model_checkpoint_path}.")
        self._logger.info(f"Training parameters: {self._training_parameters}.")
        self._logger.info(f"Logging to: {self._run._backend.connection_string}.")
        self._model.dump(open(self._model_checkpoint_path, "wb"))

        self._L_val_min = self._model.compute_loss(
            X=self._X_val, Y_true=self._Y_val, loss_fn=self._loss_fn
        )

        if self._training_parameters.trace_logging:
            tracer = Tracer(
                run_id=self._run.id,
                model=self._model,
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
                batches_per_epoch=self._training_parameters.batches_per_epoch,
                history_max_len=self._training_parameters.history_max_len,
                warmup_epochs=self._training_parameters.warmup_epochs,
            )
            self.subscribe_to_after_training_step(self._monitor.post_batch)

        self._before_training_loop()

        with self._create_training_loop_context():
            return self._training_loop()

    def _before_training_loop(self) -> None:
        if self._training_parameters.max_restarts > 0:
            self._optimizer.snapshot()

    def _training_loop(self) -> TrainingResult:
        last_log_time = time.time()
        L_val = self._model.compute_loss(
            X=self._X_val, Y_true=self._Y_val, loss_fn=self._loss_fn
        )
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
            disable=self._training_parameters.quiet,
        ):
            with self._create_training_step_context():
                batch_indices = next(self._batcher)
                X_train_batch = self._X_train[batch_indices]
                Y_train_batch = self._Y_train[batch_indices]

                if self._transform is not None:
                    X_train_batch, self._key = self._transform(X_train_batch, self._key)

                gradient, update = self._training_step(
                    X_train_batch=X_train_batch,
                    Y_train_batch=Y_train_batch,
                )
            for handler in self._after_training_step:
                match handler(gradient, update):
                    case CheckFailed() as check:
                        logger.error(f"Check failed: {check.message}")
                        return TrainingFailed(
                            model_checkpoint_path=self._model_checkpoint_path,
                            message=check.message,
                            model_checkpoint_save_epoch=self._L_val_min_epoch,
                        )
                    case None:
                        pass
                    case never:
                        assert_never(never)

            if self._training_parameters.monotonic:
                if self._last_update is not None:
                    self._revert_training_step()

            L_batch = self._model.compute_loss(
                X=X_train_batch, Y_true=Y_train_batch, loss_fn=self._loss_fn
            )

            if self._training_parameters.monotonic:
                self._last_update = update

            if (i + 1) % self._training_parameters.batches_per_epoch == 0 and i > 0:
                L_val = self._model.compute_loss(
                    X=self._X_val, Y_true=self._Y_val, loss_fn=self._loss_fn
                )
                if L_val < self._L_val_min:
                    self._L_val_min = L_val
                    self._L_val_min_epoch = self._training_parameters.current_epoch(i)
                    self._model.dump(open(self._model_checkpoint_path, "wb"))

                if (check := self._post_epoch(L_val)) is not None:
                    return TrainingFailed(
                        model_checkpoint_path=self._model_checkpoint_path,
                        message=check.message,
                        model_checkpoint_save_epoch=self._L_val_min_epoch,
                    )

            if time.time() - last_log_time > DEFAULT_LOG_INTERVAL_SECONDS:
                if not self._training_parameters.quiet:
                    tqdm.write(
                        f"Epoch {self._training_parameters.current_epoch(i)}, Batch Loss = {L_batch}, Validation Loss = {L_val}"
                        + (
                            f", {report}"
                            if (report := self._optimizer.report()) != ""
                            else ""
                        )
                    )
                last_log_time = time.time()

        if L_val is None:
            L_val = self._compute_loss(X=self._X_val, Y_true=self._Y_val)

        self._run.log_iteration(
            epoch=self._training_parameters.num_epochs,
            batch=self._training_parameters.total_batches,
            batch_loss=L_batch,
            val_loss=L_val,
            learning_rate=self._optimizer.learning_rate,
        )
        return TrainingSuccessful(
            model_checkpoint_path=self._model_checkpoint_path,
        )

    def _training_step(
        self,
        X_train_batch: jnp.ndarray,
        Y_train_batch: jnp.ndarray,
    ) -> tuple[
        Sequence[SupportsGradientOperations],
        Sequence[SupportsGradientOperations],
    ]:
        gradient, update = self._optimizer.training_step(
            X_train_batch=X_train_batch,
            Y_train_batch=Y_train_batch,
            return_gradients=True,
        )
        if self._training_parameters.monotonic:
            self._last_update = update
        return gradient, update

    def _post_epoch(self, L_val: float) -> CheckFailed | None:
        if not self._training_parameters.no_monitoring and self._monitor is not None:
            return self._monitor.post_epoch(L_val)
        return None

    def shutdown(self) -> None:
        pass
