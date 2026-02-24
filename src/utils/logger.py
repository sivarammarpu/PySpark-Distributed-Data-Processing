"""
src/utils/logger.py
===================
Structured logger that mirrors Log4j's output format used by Spark itself.

Usage
-----
    from src.utils.logger import get_logger, log_execution_time

    logger = get_logger(__name__)
    logger.info("Pipeline started")

    @log_execution_time
    def my_function(): ...
"""

import functools
import logging
import sys
import time
from typing import Callable


# ---------------------------------------------------------------------------
# Log format matching Log4j pattern: date level class - message
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger wired to stdout with a Log4j-style format.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when the module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Avoid propagation to root logger (prevents double-printing)
    logger.propagate = False

    return logger


def log_execution_time(func: Callable = None, *, logger_name: str = None):
    """
    Decorator that logs the execution time of a function.

    Can be used with or without arguments:

        @log_execution_time
        def my_func(): ...

        @log_execution_time(logger_name="my.module")
        def my_func(): ...
    """
    def decorator(fn: Callable) -> Callable:
        _logger = get_logger(logger_name or fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _logger.info("▶  Starting  : %s", fn.__qualname__)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _logger.info(
                    "✔  Completed : %s  [%.3f s]", fn.__qualname__, elapsed
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                _logger.error(
                    "✘  Failed    : %s  [%.3f s]  —  %s",
                    fn.__qualname__, elapsed, exc,
                )
                raise

        return wrapper

    # Support bare @log_execution_time (without call)
    if func is not None:
        return decorator(func)
    return decorator
