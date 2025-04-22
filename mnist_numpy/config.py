from pydantic import BaseModel


class TrainingParameters(BaseModel):
    batch_size: int
    dropout_keep_prob: tuple[float, ...]
    learning_rate_limits: tuple[float, float]
    num_epochs: int
    regulariser_lambda: float
    total_epochs: int
    trace_logging: bool
    train_set_size: int
    warmup_epochs: int
    history_max_len: int = 100
    workers: int = 0

    @property
    def batches_per_epoch(self) -> int:
        return self.train_set_size // self.batch_size

    @property
    def start_epoch(self) -> int:
        return self.total_epochs - self.num_epochs

    @property
    def start_batch(self) -> int:
        return self.start_epoch * self.batches_per_epoch

    @property
    def total_batches(self) -> int:
        return self.total_epochs * self.batches_per_epoch

    @property
    def warmup_batches(self) -> int:
        return self.warmup_epochs * self.batches_per_epoch

    def current_epoch(self, iteration: int) -> int:
        return iteration // self.batches_per_epoch

    def current_progress(self, iteration: int) -> float:
        return iteration / self.total_batches
