"""
Logging and monitoring utilities for AI-Robot Brain
Provides standardized logging and performance monitoring across all modules
"""
import logging
import time
import functools
from typing import Callable, Any, Dict
from pathlib import Path
import json
import atexit


class PerformanceMonitor:
    """
    Performance monitoring system for tracking execution times and resource usage
    """
    def __init__(self):
        self.metrics = {}
        self.timers = {}
        self._start_time = time.time()

    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]

            # Store metrics
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(elapsed)

            return elapsed
        return 0.0

    def get_average_time(self, name: str) -> float:
        """Get average execution time for a named operation"""
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return 0.0

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics"""
        summary = {}
        for name, times in self.metrics.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        return summary

    def reset(self):
        """Reset all collected metrics"""
        self.metrics = {}
        self.timers = {}
        self._start_time = time.time()


class Logger:
    """
    Standardized logging system for the AI-Robot Brain
    """
    def __init__(self, name: str = "AI-Robot-Brain", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Avoid adding multiple handlers if logger already has handlers
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper()))

            # Create file handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{name.lower().replace('-', '_')}.log")
            file_handler.setLevel(getattr(logging, level.upper()))

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers to logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

        self.performance_monitor = PerformanceMonitor()

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

    def log_function_call(self, func: Callable) -> Callable:
        """Decorator to log function calls with execution time"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.info(f"Calling function: {func.__name__}")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.info(f"Function {func.__name__} completed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.error(f"Function {func.__name__} failed after {execution_time:.4f}s: {str(e)}")
                raise
        return wrapper

    def time_block(self, name: str):
        """Context manager to time a block of code"""
        class Timer:
            def __enter__(timer_self):
                self.performance_monitor.start_timer(name)
                return timer_self

            def __exit__(timer_self, exc_type, exc_val, exc_tb):
                elapsed = self.performance_monitor.stop_timer(name)
                if exc_type is None:
                    self.info(f"Block '{name}' completed in {elapsed:.4f}s")
                else:
                    self.error(f"Block '{name}' failed after {elapsed:.4f}s: {exc_val}")

        return Timer()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return self.performance_monitor.get_metrics_summary()

    def log_performance_metrics(self):
        """Log all collected performance metrics"""
        summary = self.get_performance_summary()
        self.info("Performance Metrics Summary:")
        for name, metrics in summary.items():
            self.info(f"  {name}: avg={metrics['average_time']:.4f}s, "
                     f"count={metrics['count']}, "
                     f"total={metrics['total_time']:.4f}s")


# Global logger instance
_logger = None


def get_logger(name: str = "AI-Robot-Brain") -> Logger:
    """
    Get the global logger instance

    Args:
        name: Name for the logger (defaults to "AI-Robot-Brain")

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        from .config.config_manager import get_config
        level = get_config("common.logging.level", "INFO")
        _logger = Logger(name, level)

        # Register cleanup function to log performance metrics on exit
        def cleanup():
            if _logger:
                _logger.log_performance_metrics()

        atexit.register(cleanup)

    return _logger


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls using the global logger

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    logger = get_logger()
    return logger.log_function_call(func)


def time_block(name: str):
    """
    Context manager to time a block of code using the global logger

    Args:
        name: Name for the timed block

    Returns:
        Context manager
    """
    logger = get_logger()
    return logger.time_block(name)


# Initialize the logger when module is imported
def _initialize_logger():
    """Initialize the global logger when the module is imported"""
    global _logger
    if _logger is None:
        _logger = Logger()


# Call initialization
_initialize_logger()