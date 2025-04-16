import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Final, Self, cast

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.optimizer import OptimizerBase, OptimizerConfigT
from mnist_numpy.trainer.tracer import PerEpochTracerStrategy, Tracer, TracerConfig

ABORT_TRAINING_THRESHOLD: Final[float] = -np.log(0.1)
ABORT_TRAINING_STD_THRESHOLD: Final[float] = 0.001
DEFAULT_LOG_INTERVAL_SECONDS: Final[int] = 10


class TrainingParameters(BaseModel):
    batch_size: int
    dropout_keep_prob: tuple[float, ...]
    learning_rate: float
    learning_rate_limits: tuple[float, float]
    num_epochs: int
    regulariser_lambda: float
    total_epochs: int
    trace_logging: bool


class Batcher:
    def __init__(self, X: np.ndarray, Y: np.ndarray, batch_size: int):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.train_set_size = X.shape[0]
        self.iterations_per_epoch = self.train_set_size / self.batch_size
        self.iterations = 0
        self._shuffle()
        self._internal_iterator = zip(
            iter(np.array_split(self.X, self.train_set_size / self.batch_size)),
            iter(np.array_split(self.Y, self.train_set_size / self.batch_size)),
        )

    def _shuffle(self) -> None:
        permutation = np.random.permutation(self.train_set_size)
        self.X = self.X[permutation]
        self.Y = self.Y[permutation]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.iterations % self.iterations_per_epoch == 0:
            self._shuffle()
            self._internal_iterator = zip(
                iter(np.array_split(self.X, self.train_set_size / self.batch_size)),
                iter(np.array_split(self.Y, self.train_set_size / self.batch_size)),
            )
        self.iterations += 1
        return next(self._internal_iterator)


class ModelTrainer:
    @staticmethod
    def train(
        *,
        model: ModelT,
        optimizer: OptimizerBase[ModelT, OptimizerConfigT],
        training_parameters: TrainingParameters,
        training_log_path: Path,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> Path:
        if not training_log_path.exists():
            training_log = pd.DataFrame(
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
            training_log.to_csv(training_log_path, index=False)
        else:
            training_log = pd.read_csv(training_log_path)

        logger.info(
            f"Training model {model.__class__.__name__}"
            f" for {training_parameters.num_epochs=} iterations with {training_parameters.learning_rate=}"
            f" using optimizer {optimizer.__class__.__name__}."
        )

        model_checkpoint_path = training_log_path.with_name(
            training_log_path.name.replace("training_log.csv", "partial.pkl")
        )
        model_training_parameters_path = training_log_path.with_name(
            training_log_path.name.replace(
                "training_log.csv", "training_parameters.json"
            )
        )
        if not model_training_parameters_path.exists():
            model_training_parameters_path.write_text(
                training_parameters.model_dump_json()
            )
        else:
            training_parameters = TrainingParameters.model_validate_json(
                model_training_parameters_path.read_text()
            )

        logger.info(
            f"Training model..\nSaving partial results to: {model_checkpoint_path}."
        )
        logger.info(f"\n{training_parameters=}.")
        logger.info(f"\n{training_log_path=}.")
        model.dump(open(model_checkpoint_path, "wb"))

        start_epoch = training_parameters.total_epochs - training_parameters.num_epochs

        L_train_min = model.compute_loss(X=X_train, Y_true=Y_train)
        L_test_min = model.compute_loss(X=X_test, Y_true=Y_test)

        L_train_history: deque[float] = deque(maxlen=100)

        batcher = Batcher(X_train, Y_train, training_parameters.batch_size)
        train_set_size = X_train.shape[0]
        batches_per_epoch = train_set_size // training_parameters.batch_size
        last_log_time = time.time()
        log_interval_seconds = DEFAULT_LOG_INTERVAL_SECONDS

        if training_parameters.trace_logging:
            tracer = Tracer(
                model=cast(MultiLayerPerceptron, model),  # TODO: Fix-types
                training_log_path=training_log_path,
                tracer_config=TracerConfig(
                    trace_strategy=PerEpochTracerStrategy(
                        training_set_size=train_set_size,
                        batch_size=training_parameters.batch_size,
                    ),
                ),
            )
            optimizer.register_after_training_step_handler(tracer)

        for i in tqdm(
            range(
                start_epoch * batches_per_epoch,
                training_parameters.total_epochs * batches_per_epoch,
            ),
            initial=start_epoch * batches_per_epoch,
            total=training_parameters.total_epochs * batches_per_epoch,
        ):
            X_train_batch, Y_train_batch = next(batcher)

            optimizer.training_step(model, X_train_batch, Y_train_batch)

            if i % (train_set_size // training_parameters.batch_size) == 0:
                L_train = model.compute_loss(X=X_train, Y_true=Y_train)
                L_train_history.append(L_train)
                if len(L_train_history) == L_train_history.maxlen:
                    std_loss = np.std(L_train_history)
                    if (
                        L_train_min > ABORT_TRAINING_THRESHOLD
                        and std_loss < ABORT_TRAINING_STD_THRESHOLD
                    ):
                        raise RuntimeError("Aborting training. Model is not learning.")
                L_test = model.compute_loss(X=X_test, Y_true=Y_test)
                epoch = i // (train_set_size // training_parameters.batch_size)

                L_train_min = min(L_train_min, L_train)
                if L_test < L_test_min:
                    model.dump(open(model_checkpoint_path, "wb"))
                    L_test_min = L_test

                pd.DataFrame(
                    [
                        [
                            epoch,
                            L_train,
                            L_train_min,
                            L_test,
                            L_test_min,
                            optimizer.learning_rate,
                            datetime.now(),
                        ]
                    ],
                    columns=training_log.columns,
                ).to_csv(training_log_path, mode="a", header=False, index=False)

            if time.time() - last_log_time > log_interval_seconds:
                tqdm.write(
                    f"Iteration {i}, Epoch {epoch}, Training Loss = {L_train}, Test Loss = {L_test}"
                    + (f", {report}" if (report := optimizer.report()) != "" else "")
                )
                last_log_time = time.time()

        return model_checkpoint_path
