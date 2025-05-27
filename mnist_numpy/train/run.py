from datetime import datetime

from mnist_numpy.train.server.backends import Backend


class Run:
    def __init__(
        self,
        *,
        seed: int,
        name: str | None = None,
        started_at: datetime | None = None,
        backend: Backend,
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
            raise ValueError("Run not started")
        return self._run_id

    @property
    def seed(self) -> int:
        return self._seed

    def log_iteration(
        self,
        *,
        batch_loss: float,
        batch: int,
        epoch: int,
        learning_rate: float,
        test_loss: float,
    ) -> None:
        self._backend.log_iteration(
            batch_loss=batch_loss,
            batch=batch,
            epoch=epoch,
            learning_rate=learning_rate,
            test_loss=test_loss,
            timestamp=datetime.now(),
        )

    def log_training_parameters(self, *, training_parameters: str) -> None:
        self._backend.log_training_parameters(training_parameters=training_parameters)

    def start_run(self) -> None:
        self._backend.create()
        self._run_id = self._backend.start_run()

    def end_run(self) -> None:
        self._backend.end_run(run_id=self._run_id)
