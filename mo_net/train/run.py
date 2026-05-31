from collections.abc import Mapping
from datetime import datetime

from mo_net.train.backends.log import LoggingBackend


class TrainingRun:
    """Thin wrapper over a :class:`LoggingBackend` for one run.

    Carries the run's name + seed, the backend handle, and (once
    ``start_run`` is called) the assigned ``run_id`` and ``lineage_id``.
    Lineage is set by the caller — when resuming, pass ``lineage_id`` and
    ``parent_run_id`` from the prior run's manifest so the backend writes
    them onto the new row.
    """

    def __init__(
        self,
        *,
        seed: int,
        name: str | None = None,
        started_at: datetime | None = None,
        backend: LoggingBackend,
    ) -> None:
        if name is None:
            name = str(seed)
        self._name = name
        self._seed = seed
        self._run_id: int | None = None
        self._lineage_id: str | None = None
        self._started_at = started_at if started_at is not None else datetime.now()
        self._backend = backend

    @property
    def id(self) -> int:
        if self._run_id is None:
            raise ValueError("No run id. Call start_run() first.")
        return self._run_id

    @property
    def lineage_id(self) -> str:
        if self._lineage_id is None:
            raise ValueError("No lineage id. Call start_run() first.")
        return self._lineage_id

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_started(self) -> bool:
        return self._run_id is not None

    def log_iteration(
        self,
        *,
        epoch: int | None,
        step: int,
        metrics: Mapping[str, float],
    ) -> None:
        self._backend.log_iteration(
            run_id=self.id,
            lineage_id=self.lineage_id,
            epoch=epoch,
            step=step,
            timestamp=datetime.now(),
            metrics=metrics,
        )

    def start_run(
        self,
        *,
        total_batches: int,
        total_epochs: int,
        lineage_id: str | None = None,
        parent_run_id: int | None = None,
        build_rev: str | None = None,
    ) -> None:
        self._backend.create()
        handle = self._backend.start_run(
            name=self._name,
            seed=self._seed,
            total_batches=total_batches,
            total_epochs=total_epochs,
            lineage_id=lineage_id,
            parent_run_id=parent_run_id,
            build_rev=build_rev,
        )
        self._run_id = handle.run_id
        self._lineage_id = handle.lineage_id

    def end_run(self) -> None:
        if self._run_id is None:
            raise ValueError("Cannot end run. Call start_run() first.")
        self._backend.end_run(run_id=self._run_id)

    def update_totals(self, *, total_batches: int) -> None:
        """Back-fill ``total_batches`` after registering the run upfront
        with a placeholder. No-op if the run isn't started yet."""
        if self._run_id is None:
            return
        self._backend.update_run_totals(self._run_id, total_batches=total_batches)
