"""
Logging utilities and decorators
"""
import sys
import time
from functools import wraps
from typing import Callable, TypeVar, Any
import logging

F = TypeVar('F', bound=Callable[..., Any])


def init_worker_logging():
    """
    Initialize logging for worker processes in multiprocessing Pool.
    
    This function should be used as the initializer parameter when creating
    multiprocessing.Pool instances to ensure worker processes have proper
    logging configuration.
    
    Example:
        >>> import multiprocessing
        >>> from dragon3_pipelines.utils import init_worker_logging
        >>> with multiprocessing.Pool(processes=4, initializer=init_worker_logging) as pool:
        ...     results = pool.map(some_function, data)
    """
    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def log_time(logger_name: str) -> Callable[[F], F]:
    """
    Decorator to log execution time of a function.
    
    Args:
        logger_name: Name of the logger to use. Typically __name__ to use the 
                     module's logger, or any other valid logger name string 
                     (e.g., "test", "myapp.module").
        
    Returns:
        Decorated function that logs start time, end time, and duration
        
    Example:
        >>> @log_time(__name__)  # Use module's logger
        ... def my_function():
        ...     pass
        
        >>> @log_time("custom.logger")  # Use custom logger name
        ... def another_function():
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
