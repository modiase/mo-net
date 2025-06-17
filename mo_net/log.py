import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from functools import wraps
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


@contextmanager
def log_time(template: str, *, log_level: LogLevel = LogLevel.TRACE) -> Iterator[float]:
    start_time = time.perf_counter()
    yield start_time
    end_time = time.perf_counter()
    logger.log(log_level.value, template.format(time_taken=end_time - start_time))
