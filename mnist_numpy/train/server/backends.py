from datetime import datetime
from pathlib import Path
from typing import IO, Protocol

import pandas as pd


class Backend(Protocol):
    @property
    def connection_string(self) -> str: ...

    def create(self) -> None:
        """Create any connections or files."""

    def start_run(self) -> str:
        """Create a new run using backend."""

    def end_run(self, run_id: str) -> None:
        """End the run using backend."""

    def teardown(self) -> None:
        """Teardown any connections or files."""

    def log_training_parameters(self, *, training_parameters: str) -> None:
        """Log the training parameters."""

    def log_iteration(
        self,
        *,
        batch_loss: float,
        batch: int,
        epoch: int,
        learning_rate: float,
        timestamp: datetime,
    ) -> None: ...


class CsvBackend(Backend):
    def __init__(self, *, path: Path) -> None:
        self._path = path
        self._columns = [
            "batch_loss",
            "test_loss",
            "batch",
            "epoch",
            "learning_rate",
            "timestamp",
        ]
        self._file: IO[str] | None = None

    @property
    def connection_string(self) -> str:
        return str(self._path)

    def create(self) -> None:
        self._file = open(self._path, "w")

    def start_run(self) -> str:
        pd.DataFrame(columns=self._columns).to_csv(self._file, index=False)
        self._file.flush()
        return str(self._path)

    def end_run(self, run_id: str) -> None:
        del run_id  # unused

    def teardown(self) -> None:
        self._file.close()

    def log_iteration(
        self,
        *,
        batch_loss: float,
        test_loss: float,
        batch: int,
        epoch: int,
        learning_rate: float,
        timestamp: datetime,
    ) -> None:
        pd.DataFrame(
            {
                "batch_loss": batch_loss,
                "test_loss": test_loss,
                "batch": batch,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "timestamp": timestamp,
            },
            index=[0],
        ).to_csv(self._file, index=False, header=False)
        self._file.flush()

    def log_training_parameters(self, *, training_parameters: str) -> None:
        self._path.with_suffix(".json").write_text(training_parameters)
