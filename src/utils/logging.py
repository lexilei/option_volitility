"""Logging configuration using Loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: str | None = None,
) -> None:
    """Configure logging with Loguru.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        rotation: Log rotation size
        retention: Log retention period
        format_string: Custom format string
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured with level={level}")


def get_logger(name: str | None = None) -> "logger":
    """Get a logger instance.

    Args:
        name: Logger name (for filtering)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class LogContext:
    """Context manager for temporary log settings."""

    def __init__(self, level: str = "DEBUG"):
        """Initialize the context.

        Args:
            level: Temporary log level
        """
        self.level = level
        self.handler_id = None

    def __enter__(self) -> "LogContext":
        """Enter the context."""
        self.handler_id = logger.add(
            sys.stderr,
            level=self.level,
            format="<level>{level: <8}</level> | {message}",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context."""
        if self.handler_id is not None:
            logger.remove(self.handler_id)


def log_function_call(func):
    """Decorator to log function calls.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned successfully")
            return result
        except Exception as e:
            logger.exception(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper


def log_time(func):
    """Decorator to log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapper
