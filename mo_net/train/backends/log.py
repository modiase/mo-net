from datetime import datetime
from pathlib import Path
from typing import IO, Protocol
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from mo_net.train.backends.models import DB_PATH, DbRun, Iteration


class LoggingBackend(Protocol):
    @property
    def connection_string(self) -> str: ...

    def create(self) -> None:
        """Create any connections or files."""

    def start_run(self, seed: int, total_batches: int, total_epochs: int) -> str:
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
        val_loss: float,
        batch: int,
        epoch: int,
        learning_rate: float,
        timestamp: datetime,
    ) -> None: ...


class CsvBackend(LoggingBackend):
    def __init__(self, *, path: Path) -> None:
        self._path = path.resolve()
        self._columns = [
            "batch_loss",
            "val_loss",
            "batch",
            "epoch",
            "learning_rate",
            "timestamp",
        ]
        self._file: IO[str] | None = None

    @property
    def connection_string(self) -> str:
        return f"csv://{str(self._path)}"

    def create(self) -> None:
        self._file = open(self._path, "w")

    def start_run(self, seed: int, total_batches: int, total_epochs: int) -> str:
        del seed, total_batches, total_epochs  # unused
        pd.DataFrame(columns=self._columns).to_csv(self._file, index=False)
        if self._file is not None:
            self._file.flush()
        return str(self._path.name.replace(self._path.suffix, ""))

    def end_run(self, run_id: str) -> None:
        del run_id  # unused

    def teardown(self) -> None:
        if self._file is not None:
            self._file.close()

    def log_iteration(
        self,
        *,
        batch_loss: float,
        val_loss: float,
        batch: int,
        epoch: int,
        learning_rate: float,
        timestamp: datetime,
    ) -> None:
        pd.DataFrame(
            {
                "batch_loss": batch_loss,
                "val_loss": val_loss,
                "batch": batch,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "timestamp": timestamp,
            },
            index=[0],
        ).to_csv(self._file, index=False, header=False)
        if self._file is not None:
            self._file.flush()

    def log_training_parameters(self, *, training_parameters: str) -> None:
        self._path.with_suffix(".json").write_text(training_parameters)


class SqliteBackend(LoggingBackend):
    def __init__(self, *, path: Path | None = None) -> None:
        self._path = path if path is not None else DB_PATH
        self._session: Session | None = None
        self._current_run: DbRun | None = None
        self._engine = create_engine(f"sqlite:///{self._path}")
        self._session_maker = sessionmaker(bind=self._engine)

    @property
    def connection_string(self) -> str:
        return f"sqlite:///{self._path}"

    def create(self) -> None:
        self._session = self._session_maker()

    def start_run(self, seed: int, total_batches: int, total_epochs: int) -> str:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")

        run = DbRun.create(
            seed=seed,
            total_batches=total_batches,
            total_epochs=total_epochs,
            started_at=datetime.now(),
        )
        self._session.add(run)
        self._session.commit()
        self._current_run = run
        return str(run.id)

    def end_run(self, run_id: str) -> None:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")

        if run := self._session.get(DbRun, int(run_id)):
            run.completed_at = datetime.now()
            self._session.commit()
        self._current_run = None

    def teardown(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    def log_training_parameters(self, *, training_parameters: str) -> None:
        self._path.with_suffix(".json").write_text(training_parameters)

    def log_iteration(
        self,
        *,
        batch_loss: float,
        val_loss: float,
        batch: int,
        epoch: int,
        learning_rate: float,
        timestamp: datetime,
    ) -> None:
        if not self._session or not self._current_run:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.current_batch = batch
        self._current_run.current_batch_loss = batch_loss
        self._current_run.current_epoch = epoch
        self._current_run.current_learning_rate = learning_rate
        self._current_run.current_val_loss = val_loss
        self._current_run.current_timestamp = timestamp
        self._current_run.updated_at = timestamp

        self._session.add(
            Iteration(
                run_id=self._current_run.id,
                batch_loss=batch_loss,
                batch=batch,
                epoch=epoch,
                learning_rate=learning_rate,
                timestamp=timestamp,
                val_loss=val_loss,
            )
        )
        self._session.commit()

    def get_run(self, run_id: int) -> DbRun | None:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")
        return self._session.get(DbRun, run_id)

    def get_run_iterations(self, run_id: int) -> list[Iteration]:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")
        return (
            self._session.query(Iteration)
            .filter(Iteration.run_id == run_id)
            .order_by(Iteration.timestamp)
            .all()
        )


class NullBackend(LoggingBackend):
    def __init__(self) -> None:
        pass

    @property
    def connection_string(self) -> str:
        return "null://"

    def create(self) -> None:
        pass

    def start_run(self, seed: int, total_batches: int, total_epochs: int) -> str:
        del seed, total_batches, total_epochs  # unused
        return "-1"

    def end_run(self, run_id: str) -> None:
        del run_id  # unused

    def teardown(self) -> None:
        pass

    def log_training_parameters(self, *, training_parameters: str) -> None:
        del training_parameters  # unused

    def log_iteration(
        self,
        *,
        batch_loss: float,
        val_loss: float,
        batch: int,
        epoch: int,
        learning_rate: float,
        timestamp: datetime,
    ) -> None:
        del batch_loss, val_loss, batch, epoch, learning_rate, timestamp  # unused


def parse_connection_string(connection_string: str) -> LoggingBackend:
    match url := urlparse(connection_string):
        case url if url.scheme == "null":
            return NullBackend()
        case url if url.scheme == "csv":
            return CsvBackend(path=Path(url.path))
        case url if url.scheme == "sqlite":
            return SqliteBackend(path=Path(url.path))
        case _:
            return SqliteBackend()
