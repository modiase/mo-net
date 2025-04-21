from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True, kw_only=True)
class TrainingContext:
    training_progress: float
    model_checkpoint_path: Path | None

    @classmethod
    def default(cls) -> Self:
        return cls(training_progress=0.0, model_checkpoint_path=None)


training_context: ContextVar[TrainingContext] = ContextVar(
    "training_context", default=TrainingContext.default()
)


@contextmanager
def set_training_progress(value: float) -> Iterator[None]:
    training_context_copy = (
        cpy
        if (cpy := training_context.get()) is not None
        else TrainingContext.default()
    )
    training_context.set(
        TrainingContext(**asdict(training_context_copy) | {"training_progress": value})
    )
    try:
        yield
    finally:
        training_context.set(training_context_copy)
