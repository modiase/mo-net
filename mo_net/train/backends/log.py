import threading
from collections.abc import MutableSequence, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Protocol
from urllib.parse import urlparse

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from mo_net.settings import get_settings
from mo_net.train.backends.models import Base, DbRun, Iteration


@dataclass(frozen=True, slots=True, kw_only=True)
class LogEntry:
    batch_loss: float
    val_loss: float
    batch: int
    epoch: int
    learning_rate: float
    timestamp: datetime


class LoggingBackend(Protocol):
    @property
    def connection_string(self) -> str: ...

    def create(self) -> None:
        """Create any connections or files."""

    def start_run(
        self,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
    ) -> str:
        """Create a new run using backend."""
        ...

    def end_run(self, run_id: str) -> None:
        """End the run using backend."""

    def teardown(self) -> None:
        """Teardown any connections or files."""

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

    def start_run(
        self,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
    ) -> str:
        del name, seed, total_batches, total_epochs  # unused
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


class SqlBackend(LoggingBackend):
    """SQLAlchemy-backed logger. Subclassed for dialect-specific conveniences."""

    def __init__(
        self,
        *,
        url: str,
        batch_size: int = 10,
        max_queue_size: int = 1000,
    ) -> None:
        self._url = url
        self._session: Session | None = None
        self._current_run: DbRun | None = None
        self._engine = create_engine(url)
        self._session_maker = sessionmaker(bind=self._engine)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="logging")
        self._lock = threading.Lock()
        self._batch_size = batch_size
        self._max_queue_size = max_queue_size
        self._pending_entries: MutableSequence[LogEntry] = []

    @property
    def connection_string(self) -> str:
        return self._url

    def create(self) -> None:
        # Concurrent workers racing CREATE TABLE IF NOT EXISTS on the same
        # empty database can both pass the existence check and then collide on
        # pg_type_typname_nsp_index; treat that race as success.
        try:
            Base.metadata.create_all(self._engine)
        except (IntegrityError, OperationalError, ProgrammingError) as e:
            if "already exists" not in str(e).lower():
                raise
        self._session = self._session_maker()

    def start_run(
        self,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
    ) -> str:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")

        run = DbRun.create(
            name=name,
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
            run.current_epoch = run.total_epochs
            run.current_batch = run.total_batches
            self._session.commit()
        self._current_run = None

    def teardown(self) -> None:
        if self._current_run is not None and self._pending_entries:
            with self._lock:
                self._flush_batch(self._pending_entries, run_id=self._current_run.id)
                self._pending_entries.clear()

        if self._session:
            self._session.close()
            self._session = None
        self._executor.shutdown(wait=True)

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

        if len(self._pending_entries) >= self._max_queue_size:
            logger.warning("Log queue is full. Dropping log entry.")
            return

        self._executor.submit(
            self._log_iteration_sync,
            run_id=self._current_run.id,
            entry=LogEntry(
                batch_loss=batch_loss,
                val_loss=val_loss,
                batch=batch,
                epoch=epoch,
                learning_rate=learning_rate,
                timestamp=timestamp,
            ),
        )

    def _log_iteration_sync(self, *, run_id: int, entry: LogEntry) -> None:
        try:
            with self._lock:
                self._pending_entries.append(entry)
                if len(self._pending_entries) >= self._batch_size:
                    self._flush_batch(self._pending_entries, run_id=run_id)
                    self._pending_entries.clear()
        except Exception as e:
            logger.error(f"Error logging iteration: {e}")

    def _flush_batch(self, entries: Sequence[LogEntry], run_id: int) -> None:
        try:
            with self._session_maker() as session:
                if run := session.get(DbRun, run_id):
                    latest = entries[-1]
                    run.current_batch = latest.batch
                    run.current_batch_loss = latest.batch_loss
                    run.current_epoch = latest.epoch
                    run.current_learning_rate = latest.learning_rate
                    run.current_val_loss = latest.val_loss
                    run.current_timestamp = latest.timestamp
                    run.updated_at = latest.timestamp

                    for entry in entries:
                        session.add(
                            Iteration(
                                run_id=run.id,
                                batch_loss=entry.batch_loss,
                                batch=entry.batch,
                                epoch=entry.epoch,
                                learning_rate=entry.learning_rate,
                                timestamp=entry.timestamp,
                                val_loss=entry.val_loss,
                            )
                        )

                    session.commit()
        except Exception as e:
            logger.error(f"Error flushing batch: {e}")

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


class SqliteBackend(SqlBackend):
    """SqlBackend pointing at a local sqlite file (settings-aware default)."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        batch_size: int = 10,
        max_queue_size: int = 1000,
    ) -> None:
        self._path = (
            path if path is not None else get_settings().resolved_db_path
        ).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(
            url=f"sqlite:///{self._path}",
            batch_size=batch_size,
            max_queue_size=max_queue_size,
        )


class MariaDbBackend(SqlBackend):
    """SqlBackend pointing at MariaDB/MySQL via pymysql."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 3306,
        database: str = "mo_net",
        user: str = "mo_net",
        password: str | None = None,
        batch_size: int = 10,
        max_queue_size: int = 1000,
    ) -> None:
        auth = f"{user}:{password}" if password else user
        super().__init__(
            url=f"mysql+pymysql://{auth}@{host}:{port}/{database}",
            batch_size=batch_size,
            max_queue_size=max_queue_size,
        )


class PostgresBackend(SqlBackend):
    """SqlBackend pointing at PostgreSQL via psycopg (v3)."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 5432,
        database: str = "mo_net",
        user: str = "mo_net",
        password: str | None = None,
        batch_size: int = 10,
        max_queue_size: int = 1000,
    ) -> None:
        auth = f"{user}:{password}" if password else user
        super().__init__(
            url=f"postgresql+psycopg://{auth}@{host}:{port}/{database}",
            batch_size=batch_size,
            max_queue_size=max_queue_size,
        )


class InMemorySqliteBackend(LoggingBackend):
    """SQLite backend using in-memory database with auto-created tables."""

    def __init__(self) -> None:
        self._engine = create_engine("sqlite:///:memory:")
        self._session_maker = sessionmaker(bind=self._engine)
        self._session: Session | None = None
        self._current_run: DbRun | None = None

    @property
    def connection_string(self) -> str:
        return "sqlite://:memory:"

    def create(self) -> None:
        Base.metadata.create_all(self._engine)
        self._session = self._session_maker()

    def start_run(
        self,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
    ) -> str:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")

        run = DbRun.create(
            name=name,
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
            run.current_epoch = run.total_epochs
            run.current_batch = run.total_batches
            self._session.commit()
        self._current_run = None

    def teardown(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

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
        self._current_run.current_batch = batch
        self._current_run.current_batch_loss = batch_loss
        self._current_run.current_epoch = epoch
        self._current_run.current_learning_rate = learning_rate
        self._current_run.current_val_loss = val_loss
        self._current_run.current_timestamp = timestamp
        self._current_run.updated_at = timestamp
        self._session.commit()


class NullBackend(LoggingBackend):
    def __init__(self) -> None:
        pass

    @property
    def connection_string(self) -> str:
        return "null://"

    def create(self) -> None:
        pass

    def start_run(
        self,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
    ) -> str:
        del name, seed, total_batches, total_epochs  # unused
        return "-1"

    def end_run(self, run_id: str) -> None:
        del run_id  # unused

    def teardown(self) -> None:
        pass

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
        case url if url.scheme in ("mysql", "mariadb", "mysql+pymysql"):
            # Force the pymysql driver to avoid pulling C-extension mysqlclient.
            normalised = (
                connection_string
                if "+pymysql" in url.scheme
                else connection_string.replace(
                    f"{url.scheme}://", "mysql+pymysql://", 1
                )
            )
            return SqlBackend(url=normalised)
        case url if url.scheme in (
            "postgres",
            "postgresql",
            "postgresql+psycopg",
        ):
            # Force the psycopg (v3) driver — the default for "postgresql://"
            # is psycopg2 which we don't ship.
            normalised = (
                connection_string
                if "+psycopg" in url.scheme
                else connection_string.replace(
                    f"{url.scheme}://", "postgresql+psycopg://", 1
                )
            )
            return SqlBackend(url=normalised)
        case _:
            return SqliteBackend()
