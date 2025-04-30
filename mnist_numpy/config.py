from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator


class TrainingParameters(BaseModel):
    batch_size: int
    dropout_keep_probs: tuple[float, ...]
    history_max_len: int
    learning_rate_limits: tuple[float, float]
    log_path: Path
    monotonic: bool
    no_monitoring: bool
    no_transform: bool
    num_epochs: int
    regulariser_lambda: float
    trace_logging: bool
    train_set_size: int
    warmup_epochs: int
    workers: int = 0

    @property
    def batches_per_epoch(self) -> int:
        return self.train_set_size // self.batch_size

    @property
    def total_batches(self) -> int:
        return self.num_epochs * self.batches_per_epoch

    @property
    def warmup_batches(self) -> int:
        return self.warmup_epochs * self.batches_per_epoch

    def current_epoch(self, iteration: int) -> int:
        return iteration // self.batches_per_epoch

    def current_progress(self, iteration: int) -> float:
        return iteration / self.total_batches

    @model_validator(mode="after")
    def validate_batch_size_for_monotonic(self) -> Self:
        if self.monotonic and self.batch_size != self.train_set_size:
            raise ValueError(
                "When monotonic is True, batch_size must equal train_set_size"
            )
        return self
