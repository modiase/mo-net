from collections.abc import Callable
from functools import wraps
import sys
from enum import StrEnum
from typing import ParamSpec, TypeVar

from loguru import logger


class LogLevel(StrEnum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_logging(log_level: LogLevel) -> None:
    logger.remove()
    logger.add(sys.stderr, level=log_level.value)

P = ParamSpec("P")
R = TypeVar("R")
def log_result(log_level: LogLevel) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = func(*args, **kwargs)
            logger.log(log_level.value, f"{func.__name__} returned {result}")
            return result
        return wrapper
    return decorator