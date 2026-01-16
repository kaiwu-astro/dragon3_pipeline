"""
Logging utilities and decorators
"""
import time
from functools import wraps
from typing import Callable, TypeVar, Any
import logging

F = TypeVar('F', bound=Callable[..., Any])


def log_time(logger_name: str) -> Callable[[F], F]:
    """
    Decorator to log execution time of a function.
    
    Args:
        logger_name: Name of the logger to use (e.g., __name__ or "test")
        
    Returns:
        Decorated function that logs start time, end time, and duration
        
    Example:
        >>> @log_time(__name__)
        ... def my_function():
        ...     pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Dynamically get logger at runtime to avoid import-time capture
            logger = logging.getLogger(logger_name)
            start_time = time.time()
            logger.debug(
                f"Function {func.__name__} started at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
            )
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(
                f"Function {func.__name__} finished at "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, "
                f"took {end_time - start_time:.4f} seconds"
            )
            return result
        return wrapper  # type: ignore
    return decorator
