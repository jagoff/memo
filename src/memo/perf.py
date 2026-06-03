"""Performance monitoring utilities."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

_log = logging.getLogger(__name__)


def timer(
    *,
    name: str | None = None,
    log_threshold_ms: float = 100.0,
    level: int = logging.DEBUG,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that times function execution and logs slow calls.

    Args:
        name: Optional name for the timer (defaults to function name).
        log_threshold_ms: Only log calls slower than this threshold (ms).
        level: Logging level to use (default DEBUG).

    Example:
        @timer(log_threshold_ms=50.0)
        def expensive_operation(x: int) -> int:
            return x * 2
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        timer_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if elapsed_ms >= log_threshold_ms:
                    _log.log(level, "%s took %.2fms", timer_name, elapsed_ms)

        return wrapper

    return decorator
