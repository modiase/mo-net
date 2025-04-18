from pydantic import BaseModel


class TrainingParameters(BaseModel):
    batch_size: int
    dropout_keep_prob: tuple[float, ...]
    learning_rate_limits: tuple[float, float]
    low_gradient_abort_threshold: float
    high_gradient_abort_threshold: float
    num_epochs: int
    regulariser_lambda: float
    total_epochs: int
    trace_logging: bool
    train_set_size: int
    warmup_epochs: int
    history_max_len: int = 100
