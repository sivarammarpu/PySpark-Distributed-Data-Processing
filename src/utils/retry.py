"""
src/utils/retry.py
==================
Decorator-based retry mechanism with exponential back-off.

Usage
-----
    from src.utils.retry import retry

    @retry(max_attempts=3, delay=2.0, backoff=2.0)
    def flaky_ingestion():
        ...
"""

import functools
import time
from typing import Callable, Tuple, Type

from src.utils.logger import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Retry a function up to *max_attempts* times on the given *exceptions*.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts (including the first call).
    delay : float
        Initial wait time in seconds between attempts.
    backoff : float
        Multiplier applied to *delay* after each failure.
    exceptions : tuple
        Exception types that trigger a retry.  All others propagate immediately.

    Returns
    -------
    Callable
        The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exc = None

            for attempt in range(1, max_attempts + 1):
                try:
                    if attempt > 1:
                        logger.info(
                            "Retry attempt %d/%d for '%s' after %.1f s …",
                            attempt, max_attempts, func.__qualname__, current_delay,
                        )
                    return func(*args, **kwargs)

                except exceptions as exc:
                    last_exc = exc
                    logger.warning(
                        "Attempt %d/%d failed for '%s': %s",
                        attempt, max_attempts, func.__qualname__, exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(current_delay)
                        current_delay *= backoff

            logger.error(
                "All %d attempts exhausted for '%s'. Raising last exception.",
                max_attempts, func.__qualname__,
            )
            raise last_exc

        return wrapper
    return decorator
