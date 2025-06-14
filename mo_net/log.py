import sys
from enum import StrEnum

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
