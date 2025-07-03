from datetime import datetime

from mo_net.train.backends.log import LoggingBackend


class TrainingRun:
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
        self._run_id: str | None = None
        self._started_at = started_at if started_at is not None else datetime.now()
        self._backend = backend

    @property
    def id(self) -> str:
        if self._run_id is None:
            raise ValueError("No run id. Call start_run() first.")
        return self._run_id

    @property
    def seed(self) -> int:
        return self._seed

    def log_iteration(
        self,
        *,
        batch: int,
        batch_loss: float,
        epoch: int,
        learning_rate: float,
        val_loss: float,
    ) -> None:
        self._backend.log_iteration(
            batch=batch,
            batch_loss=batch_loss,
            epoch=epoch,
            learning_rate=learning_rate,
            timestamp=datetime.now(),
            val_loss=val_loss,
        )

    def log_training_parameters(self, *, training_parameters: str) -> None:
        self._backend.log_training_parameters(training_parameters=training_parameters)

    def start_run(self, total_batches: int, total_epochs: int) -> None:
        self._backend.create()
        self._run_id = self._backend.start_run(
            name=self._name,
            seed=self._seed,
            total_batches=total_batches,
            total_epochs=total_epochs,
        )

    def end_run(self) -> None:
        if self._run_id is None:
            raise ValueError("Cannot end run. Call start_run() first.")
        self._backend.end_run(run_id=self._run_id)

    @property
    def name(self) -> str:
        return self._name
