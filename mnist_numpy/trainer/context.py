from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TypedDict


class TrainingContext(TypedDict):
    training_progress: float
    model_checkpoint_path: Path


training_context: ContextVar[TrainingContext | None] = ContextVar(
    "training_context", default=None
)


@contextmanager
def set_training_context(value: TrainingContext) -> Iterator[None]:
    training_context.set(value)
    try:
        yield
    finally:
        training_context.set(None)
