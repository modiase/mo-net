import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from mnist_numpy.functions import cross_entropy, softmax
from mnist_numpy.model.base import ModelT
from mnist_numpy.optimizer import OptimizerBase, OptimizerConfigT

DEFAULT_LOG_INTERVAL_SECONDS: Final[int] = 10


class TrainingParameters(BaseModel):
    batch_size: int
    learning_rate: float
    learning_rate_limits: tuple[float, float]
    learning_rate_rescale_factor_per_epoch: float
    momentum_parameter: float
    num_epochs: int
    total_epochs: int


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
                    "moving_average_training_loss",
                    "test_loss",
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

        train_set_size = X_train.shape[0]

        X_train_batched = iter(
            np.array_split(X_train, train_set_size // training_parameters.batch_size)
        )
        Y_train_batched = iter(
            np.array_split(Y_train, train_set_size // training_parameters.batch_size)
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

        k_train = 1 / train_set_size

        test_set_size = X_test.shape[0]
        k_test = 1 / test_set_size

        L_train_min = k_train * cross_entropy(
            softmax(model._forward_prop(X_train)[1][-1]), Y_train
        )
        L_test_min = k_test * cross_entropy(
            softmax(model._forward_prop(X_test)[1][-1]), Y_test
        )
        training_loss_history = deque([L_train_min], maxlen=10)

        last_log_time = time.time()
        log_interval_seconds = DEFAULT_LOG_INTERVAL_SECONDS
        batches_per_epoch = train_set_size // training_parameters.batch_size
        for i in tqdm(
            range(
                start_epoch * batches_per_epoch,
                training_parameters.total_epochs * batches_per_epoch,
            ),
            initial=start_epoch * batches_per_epoch,
            total=training_parameters.total_epochs * batches_per_epoch,
        ):
            X_train_batch = next(X_train_batched)
            Y_train_batch = next(Y_train_batched)

            optimizer.update(model, X_train_batch, Y_train_batch)

            if i % (train_set_size // training_parameters.batch_size) == 0:
                permutation = np.random.permutation(train_set_size)
                X_train = X_train[permutation]
                Y_train = Y_train[permutation]

                X_train_batched = iter(
                    np.array_split(
                        X_train, train_set_size // training_parameters.batch_size
                    )
                )
                Y_train_batched = iter(
                    np.array_split(
                        Y_train, train_set_size // training_parameters.batch_size
                    )
                )
                _, A_train = model._forward_prop(X_train)
                L_train = k_train * cross_entropy(softmax(A_train[-1]), Y_train)
                training_loss_history.append(L_train)
                _, A_test = model._forward_prop(X_test)
                L_test = k_test * cross_entropy(softmax(A_test[-1]), Y_test)
                epoch = i // (train_set_size // training_parameters.batch_size)
                pd.DataFrame(
                    [
                        [
                            epoch,
                            L_train,
                            L_train_min,
                            np.average(training_loss_history),
                            L_test,
                            optimizer.learning_rate,
                            datetime.now(),
                        ]
                    ],
                    columns=training_log.columns,
                ).to_csv(training_log_path, mode="a", header=False, index=False)
                L_train_min = min(L_train_min, L_train)
                if L_test < L_test_min:
                    model.dump(open(model_checkpoint_path, "wb"))
                    L_test_min = L_test
            if time.time() - last_log_time > log_interval_seconds:
                tqdm.write(
                    f"Iteration {i}, Epoch {epoch}, Training Loss = {L_train}, Test Loss = {L_test}"
                    + (f", {report}" if (report := optimizer.report()) != "" else "")
                )
                last_log_time = time.time()

        return model_checkpoint_path
