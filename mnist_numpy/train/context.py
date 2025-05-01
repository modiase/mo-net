from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True, kw_only=True)
class TrainingContext:
    training_progress: float
    model_checkpoint_path: Path | None
    model_checkpoint_save_epoch: int | None

    @classmethod
    def default(cls) -> Self:
        return cls(
            training_progress=0.0,
            model_checkpoint_path=None,
            model_checkpoint_save_epoch=None,
        )


training_context: ContextVar[TrainingContext] = ContextVar(
    "training_context", default=TrainingContext.default()
)


def set_model_checkpoint_save_epoch(epoch: int) -> None:
    training_context.set(
        TrainingContext(
            **asdict(training_context.get()) | {"model_checkpoint_save_epoch": epoch}
        )
    )


def set_training_progress(value: float) -> None:
    training_context.set(
        TrainingContext(**asdict(training_context.get()) | {"training_progress": value})
    )
