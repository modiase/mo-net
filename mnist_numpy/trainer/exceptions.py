from pathlib import Path

from mnist_numpy.trainer.context import TrainingContext, training_context


class AbortTraining(RuntimeError):
    def __init__(
        self,
        message: str,
        training_progress: float | None = None,
        model_checkpoint_path: Path | None = None,
    ):
        super().__init__(message + (" " if message else "") + "Aborting training.")
        training_context_ = (
            context
            if (context := training_context.get()) is not None
            else TrainingContext.default()
        )
        self.training_progress = (
            training_progress
            if training_progress is not None
            else training_context_.training_progress
        )
        self.model_checkpoint_path = (
            model_checkpoint_path
            if model_checkpoint_path is not None
            else training_context_.model_checkpoint_path
        )
