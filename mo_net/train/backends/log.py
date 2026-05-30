"""Training-log backends.

The :class:`LoggingBackend` protocol takes iterations with a free-form
``metrics`` mapping; SQL backends fan that mapping into the EAV
``metrics`` table. ``lineage_id`` is threaded through ``start_run`` so
resumed runs share an id with their parent and chain trajectories join
in O(1).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping, MutableSequence, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from loguru import logger
from sqlalchemy import create_engine, event, insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from mo_net.settings import get_settings
from mo_net.train.backends.models import (
    DbRun,
    Iteration,
    Metric,
    install_schema,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LogEntry:
    """One queued iteration awaiting a batched flush."""

    run_id: int
    lineage_id: str
    epoch: int | None
    step: int
    timestamp: datetime
    metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RunHandle:
    """Returned by :meth:`LoggingBackend.start_run`."""

    run_id: int
    lineage_id: str


class LoggingBackend(Protocol):
    @property
    def connection_string(self) -> str: ...

    def create(self) -> None:
        """Create any connections or tables."""

    def start_run(
        self,
        *,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
        lineage_id: str | None,
        parent_run_id: int | None,
        build_rev: str | None,
    ) -> RunHandle:
        """Start a new run; mint a ``lineage_id`` if one wasn't supplied."""
        ...

    def end_run(self, run_id: int) -> None:
        """Mark the run completed."""

    def teardown(self) -> None:
        """Flush buffers, close connections."""

    def log_iteration(
        self,
        *,
        run_id: int,
        lineage_id: str,
        epoch: int | None,
        step: int,
        timestamp: datetime,
        metrics: Mapping[str, float],
    ) -> None: ...

    def lookup_run_id_by_name(self, name: str) -> int | None:
        """Resolve a run's id by name (used to derive parent_run_id on resume).

        Returns ``None`` when the backend can't satisfy the query (e.g.
        :class:`NullBackend`) or the name isn't present.
        """
        ...


@event.listens_for(Engine, "connect")
def _enforce_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    """Enable ``PRAGMA foreign_keys = ON`` for every sqlite connection.

    Without this, ``ON DELETE CASCADE`` on the metrics→iterations FK
    silently no-ops. Other dialects ignore PRAGMA.
    """
    del connection_record  # unused
    if "sqlite3" in type(dbapi_connection).__module__:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class SqlBackend(LoggingBackend):
    """SQLAlchemy-backed logger. Subclassed for dialect-specific URLs."""

    def __init__(
        self,
        *,
        url: str,
        batch_size: int = 10,
        max_queue_size: int = 1000,
    ) -> None:
        self._url = url
        self._session: Session | None = None
        self._current_handle: RunHandle | None = None
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
            install_schema(self._engine)
        except (IntegrityError, OperationalError, ProgrammingError) as e:
            if "already exists" not in str(e).lower():
                raise
        self._session = self._session_maker()

    def start_run(
        self,
        *,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
        lineage_id: str | None,
        parent_run_id: int | None,
        build_rev: str | None,
    ) -> RunHandle:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")

        resolved_lineage = lineage_id if lineage_id is not None else str(uuid.uuid4())
        run = DbRun.create(
            name=name,
            seed=seed,
            lineage_id=resolved_lineage,
            parent_run_id=parent_run_id,
            total_batches=total_batches,
            total_epochs=total_epochs,
            build_rev=build_rev,
            started_at=datetime.now(),
        )
        self._session.add(run)
        self._session.commit()
        self._current_handle = RunHandle(run_id=run.id, lineage_id=resolved_lineage)
        return self._current_handle

    def end_run(self, run_id: int) -> None:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")

        if run := self._session.get(DbRun, run_id):
            run.completed_at = datetime.now()
            self._session.commit()
        self._current_handle = None

    def teardown(self) -> None:
        if self._current_handle is not None and self._pending_entries:
            with self._lock:
                self._flush_batch(self._pending_entries)
                self._pending_entries.clear()

        if self._session:
            self._session.close()
            self._session = None
        self._executor.shutdown(wait=True)

    def log_iteration(
        self,
        *,
        run_id: int,
        lineage_id: str,
        epoch: int | None,
        step: int,
        timestamp: datetime,
        metrics: Mapping[str, float],
    ) -> None:
        if not self._session or not self._current_handle:
            raise RuntimeError("No active run. Call start_run() first.")

        if len(self._pending_entries) >= self._max_queue_size:
            logger.warning("Log queue is full. Dropping log entry.")
            return

        entry = LogEntry(
            run_id=run_id,
            lineage_id=lineage_id,
            epoch=epoch,
            step=step,
            timestamp=timestamp,
            metrics=tuple(metrics.items()),
        )
        self._executor.submit(self._log_iteration_sync, entry=entry)

    def _log_iteration_sync(self, *, entry: LogEntry) -> None:
        try:
            with self._lock:
                self._pending_entries.append(entry)
                if len(self._pending_entries) >= self._batch_size:
                    self._flush_batch(self._pending_entries)
                    self._pending_entries.clear()
        except Exception as e:
            logger.error(f"Error logging iteration: {e}")

    def _flush_batch(self, entries: Sequence[LogEntry]) -> None:
        if not entries:
            return
        try:
            with self._session_maker() as session:
                iteration_rows = [
                    {
                        "run_id": entry.run_id,
                        "step": entry.step,
                        "epoch": entry.epoch,
                        "lineage_id": entry.lineage_id,
                        "timestamp": entry.timestamp,
                    }
                    for entry in entries
                ]
                metric_rows = [
                    {
                        "run_id": entry.run_id,
                        "step": entry.step,
                        "name": name,
                        "value": value,
                    }
                    for entry in entries
                    for name, value in entry.metrics
                ]
                # Intermediate flush lets sqlite's PRAGMA-enforced FK see
                # the iteration rows before the metric INSERTs hit it.
                session.execute(insert(Iteration), iteration_rows)
                session.flush()
                if metric_rows:
                    session.execute(insert(Metric), metric_rows)
                latest_by_run: dict[int, datetime] = {}
                for entry in entries:
                    prior = latest_by_run.get(entry.run_id)
                    if prior is None or entry.timestamp > prior:
                        latest_by_run[entry.run_id] = entry.timestamp
                for run_id, ts in latest_by_run.items():
                    if run := session.get(DbRun, run_id):
                        run.last_iteration_at = ts
                session.commit()
        except Exception as e:
            logger.error(f"Error flushing batch: {e}")

    def lookup_run_id_by_name(self, name: str) -> int | None:
        if not self._session:
            raise RuntimeError("Session not created. Call create() first.")
        run = (
            self._session.query(DbRun)
            .filter(DbRun.name == name)
            .order_by(DbRun.started_at.desc())
            .first()
        )
        return run.id if run else None

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
            .order_by(Iteration.step)
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


class InMemorySqliteBackend(SqlBackend):
    """SQLite in-memory backend with a shared connection.

    Writes go straight through the main session (no executor queue) so
    tests can read what they just wrote without waiting for a background
    flush. ``StaticPool`` keeps a single shared connection so every
    session sees the same in-memory database (each fresh
    ``sqlite:///:memory:`` connection would otherwise see an empty DB).
    """

    def __init__(self) -> None:
        self._url = "sqlite:///:memory:"
        self._session: Session | None = None
        self._current_handle: RunHandle | None = None
        self._engine = create_engine(
            self._url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self._session_maker = sessionmaker(bind=self._engine)
        # Inherited from SqlBackend but unused: this subclass writes
        # synchronously through the main session.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="logging")
        self._lock = threading.Lock()
        self._batch_size = 1
        self._max_queue_size = 10_000
        self._pending_entries: MutableSequence[LogEntry] = []

    def log_iteration(
        self,
        *,
        run_id: int,
        lineage_id: str,
        epoch: int | None,
        step: int,
        timestamp: datetime,
        metrics: Mapping[str, float],
    ) -> None:
        if not self._session or not self._current_handle:
            raise RuntimeError("No active run. Call start_run() first.")
        self._session.add(
            Iteration(
                run_id=run_id,
                step=step,
                epoch=epoch,
                lineage_id=lineage_id,
                timestamp=timestamp,
            )
        )
        self._session.flush()
        for name, value in metrics.items():
            self._session.add(Metric(run_id=run_id, step=step, name=name, value=value))
        if run := self._session.get(DbRun, run_id):
            run.last_iteration_at = timestamp
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
        *,
        name: str,
        seed: int,
        total_batches: int,
        total_epochs: int,
        lineage_id: str | None,
        parent_run_id: int | None,
        build_rev: str | None,
    ) -> RunHandle:
        del name, seed, total_batches, total_epochs, parent_run_id, build_rev
        resolved = lineage_id if lineage_id is not None else str(uuid.uuid4())
        return RunHandle(run_id=-1, lineage_id=resolved)

    def end_run(self, run_id: int) -> None:
        del run_id

    def teardown(self) -> None:
        pass

    def log_iteration(
        self,
        *,
        run_id: int,
        lineage_id: str,
        epoch: int | None,
        step: int,
        timestamp: datetime,
        metrics: Mapping[str, float],
    ) -> None:
        del run_id, lineage_id, epoch, step, timestamp, metrics

    def lookup_run_id_by_name(self, name: str) -> int | None:
        del name
        return None


def parse_connection_string(connection_string: str) -> LoggingBackend:
    match url := urlparse(connection_string):
        case url if url.scheme == "null":
            return NullBackend()
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
        case url if url.scheme in ("postgres", "postgresql", "postgresql+psycopg"):
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
