"""Tests for the tall/skinny logging backend.

Exercises every concrete `LoggingBackend` against the same scenarios so
adding a new backend reuses the parametrisation. The headline tests:
write happens, lineage is minted/inherited, the EAV `metrics` rows show
up in the `run_metrics_summary` view, and the SQLite FK cascade is
actually wired up (without the PRAGMA we set, ON DELETE CASCADE silently
no-ops).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from mo_net.train.backends.log import (
    InMemorySqliteBackend,
    LoggingBackend,
    NullBackend,
    SqliteBackend,
    parse_connection_string,
)


@pytest.fixture
def in_memory_backend() -> Iterator[InMemorySqliteBackend]:
    backend = InMemorySqliteBackend()
    backend.create()
    try:
        yield backend
    finally:
        backend.teardown()


@pytest.fixture
def sqlite_backend(tmp_path: Path) -> Iterator[SqliteBackend]:
    backend = SqliteBackend(path=tmp_path / "test.db")
    backend.create()
    try:
        yield backend
    finally:
        backend.teardown()


class TestSchemaRoundTrip:
    """Iteration + metric writes survive a round-trip through the backend."""

    def test_log_iteration_writes_iteration_and_metrics(
        self, in_memory_backend: InMemorySqliteBackend
    ) -> None:
        handle = in_memory_backend.start_run(
            name="rt",
            seed=1,
            total_batches=10,
            total_epochs=2,
            lineage_id=None,
            parent_run_id=None,
            build_rev=None,
        )
        in_memory_backend.log_iteration(
            run_id=handle.run_id,
            lineage_id=handle.lineage_id,
            epoch=0,
            step=1,
            timestamp=datetime.now(),
            metrics={"val_loss": 5.0, "batch_loss": 5.1},
        )
        session = in_memory_backend._session
        assert session is not None
        iter_rows = session.execute(
            text("SELECT run_id, step, epoch FROM iterations")
        ).fetchall()
        metric_rows = session.execute(
            text("SELECT name, value FROM metrics")
        ).fetchall()
        assert iter_rows == [(handle.run_id, 1, 0)]
        assert sorted(metric_rows) == [("batch_loss", 5.1), ("val_loss", 5.0)]


class TestLineage:
    """`lineage_id` is minted when absent, inherited when supplied,
    and `parent_run_id` is stored verbatim.
    """

    def test_lineage_id_minted_when_none(
        self, in_memory_backend: InMemorySqliteBackend
    ) -> None:
        handle = in_memory_backend.start_run(
            name="root",
            seed=1,
            total_batches=1,
            total_epochs=1,
            lineage_id=None,
            parent_run_id=None,
            build_rev=None,
        )
        assert handle.lineage_id  # truthy uuid string
        assert handle.run_id > 0

    def test_lineage_id_inherited_when_supplied(
        self, in_memory_backend: InMemorySqliteBackend
    ) -> None:
        first = in_memory_backend.start_run(
            name="seg1",
            seed=1,
            total_batches=1,
            total_epochs=1,
            lineage_id=None,
            parent_run_id=None,
            build_rev=None,
        )
        in_memory_backend.end_run(first.run_id)
        second = in_memory_backend.start_run(
            name="seg2",
            seed=2,
            total_batches=1,
            total_epochs=1,
            lineage_id=first.lineage_id,
            parent_run_id=first.run_id,
            build_rev=None,
        )
        assert second.lineage_id == first.lineage_id
        assert second.run_id != first.run_id

        session = in_memory_backend._session
        assert session is not None
        rows = session.execute(
            text("SELECT id, lineage_id, parent_run_id FROM runs ORDER BY id")
        ).fetchall()
        assert rows == [
            (first.run_id, first.lineage_id, None),
            (second.run_id, first.lineage_id, first.run_id),
        ]


class TestRunMetricsSummaryView:
    """`run_metrics_summary` returns correct min/max/latest aggregates."""

    def test_min_max_latest(self, in_memory_backend: InMemorySqliteBackend) -> None:
        handle = in_memory_backend.start_run(
            name="view",
            seed=1,
            total_batches=5,
            total_epochs=1,
            lineage_id=None,
            parent_run_id=None,
            build_rev=None,
        )
        now = datetime.now()
        for step, val in enumerate([5.0, 4.0, 3.5, 4.2, 3.8], start=1):
            in_memory_backend.log_iteration(
                run_id=handle.run_id,
                lineage_id=handle.lineage_id,
                epoch=0,
                step=step,
                timestamp=now + timedelta(seconds=step),
                metrics={"val_loss": val},
            )
        session = in_memory_backend._session
        assert session is not None
        row = session.execute(
            text(
                "SELECT min_value, max_value, latest_value, "
                "latest_step, min_value_step, n_samples "
                "FROM run_metrics_summary WHERE name = 'val_loss'"
            )
        ).fetchone()
        assert row is not None
        min_value, max_value, latest_value, latest_step, min_value_step, n_samples = row
        assert min_value == pytest.approx(3.5)
        assert max_value == pytest.approx(5.0)
        assert latest_value == pytest.approx(3.8)
        assert latest_step == 5
        assert min_value_step == 3
        assert n_samples == 5


class TestSqliteForeignKeyEnforcement:
    """`PRAGMA foreign_keys=ON` makes `ON DELETE CASCADE` actually cascade."""

    def test_deleting_run_cascades_to_iterations_and_metrics(
        self, sqlite_backend: SqliteBackend
    ) -> None:
        handle = sqlite_backend.start_run(
            name="cascade",
            seed=1,
            total_batches=2,
            total_epochs=1,
            lineage_id=None,
            parent_run_id=None,
            build_rev=None,
        )
        sqlite_backend.log_iteration(
            run_id=handle.run_id,
            lineage_id=handle.lineage_id,
            epoch=0,
            step=1,
            timestamp=datetime.now(),
            metrics={"val_loss": 5.0},
        )
        sqlite_backend.teardown()  # flushes the executor

        # Re-create the backend (fresh session) for the read + delete.
        backend2 = SqliteBackend(path=sqlite_backend._path)
        backend2.create()
        session = backend2._session
        assert session is not None
        # Pre-delete: rows exist in all three tables.
        assert session.execute(text("SELECT COUNT(*) FROM runs")).scalar() == 1
        assert session.execute(text("SELECT COUNT(*) FROM iterations")).scalar() == 1
        assert session.execute(text("SELECT COUNT(*) FROM metrics")).scalar() == 1
        # Delete the run; the cascade should clear iterations + metrics.
        session.execute(
            text("DELETE FROM runs WHERE id = :rid"), {"rid": handle.run_id}
        )
        session.commit()
        assert session.execute(text("SELECT COUNT(*) FROM iterations")).scalar() == 0
        assert session.execute(text("SELECT COUNT(*) FROM metrics")).scalar() == 0
        backend2.teardown()


class TestProtocolUniform:
    """Every concrete backend accepts the new signature without exception."""

    @pytest.mark.parametrize(
        "backend_factory",
        [
            lambda: NullBackend(),
            lambda: InMemorySqliteBackend(),
        ],
    )
    def test_full_lifecycle(self, backend_factory) -> None:
        backend: LoggingBackend = backend_factory()
        backend.create()
        handle = backend.start_run(
            name="proto",
            seed=1,
            total_batches=1,
            total_epochs=1,
            lineage_id=None,
            parent_run_id=None,
            build_rev="abc1234",
        )
        backend.log_iteration(
            run_id=handle.run_id,
            lineage_id=handle.lineage_id,
            epoch=0,
            step=1,
            timestamp=datetime.now(),
            metrics={"val_loss": 1.0, "batch_loss": 1.1, "learning_rate": 1e-4},
        )
        backend.end_run(handle.run_id)
        backend.teardown()


class TestCsvBackendRemoved:
    """`CsvBackend` is gone; `csv://` falls through to the default sqlite backend."""

    def test_csv_class_not_importable(self) -> None:
        import mo_net.train.backends.log as log_mod

        assert not hasattr(log_mod, "CsvBackend")

    def test_csv_scheme_falls_back_to_sqlite(self, tmp_path: Path) -> None:
        # parse_connection_string's default branch returns a SqliteBackend.
        backend = parse_connection_string("csv:///irrelevant/path")
        assert isinstance(backend, SqliteBackend)


class TestLookupRunIdByName:
    """`lookup_run_id_by_name` is what the CLI uses to resolve parent_run_id."""

    def test_returns_id_for_existing_run(
        self, in_memory_backend: InMemorySqliteBackend
    ) -> None:
        handle = in_memory_backend.start_run(
            name="findme",
            seed=1,
            total_batches=1,
            total_epochs=1,
            lineage_id=None,
            parent_run_id=None,
            build_rev=None,
        )
        assert in_memory_backend.lookup_run_id_by_name("findme") == handle.run_id

    def test_returns_none_for_unknown_run(
        self, in_memory_backend: InMemorySqliteBackend
    ) -> None:
        assert in_memory_backend.lookup_run_id_by_name("nope") is None

    def test_null_backend_returns_none(self) -> None:
        assert NullBackend().lookup_run_id_by_name("anything") is None
