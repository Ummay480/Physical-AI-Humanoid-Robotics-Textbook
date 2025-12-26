"""
Performance optimization utilities for real-time perception processing.

Provides tools for optimizing latency, memory usage, and computational efficiency.
"""
import time
import psutil
import threading
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
import numpy as np
from collections import deque


class LatencyOptimizer:
    """
    Optimize processing latency for real-time requirements.
    """

    def __init__(self, target_latency_ms: float = 20.0):
        """
        Initialize latency optimizer.

        Args:
            target_latency_ms: Target latency in milliseconds
        """
        self.target_latency_ms = target_latency_ms
        self.latency_history = deque(maxlen=100)

    def measure_latency(self, func: Callable) -> Callable:
        """
        Decorator to measure function latency.

        Args:
            func: Function to measure

        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000  # ms

            self.latency_history.append(duration)

            if duration > self.target_latency_ms:
                print(f"Warning: {func.__name__} exceeded target latency: "
                      f"{duration:.2f}ms > {self.target_latency_ms}ms")

            return result

        return wrapper

    def get_average_latency(self) -> float:
        """Get average latency from history."""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)

    def is_meeting_target(self) -> bool:
        """Check if average latency meets target."""
        return self.get_average_latency() <= self.target_latency_ms


class MemoryOptimizer:
    """
    Optimize memory usage for long-running processes.
    """

    def __init__(self):
        """Initialize memory optimizer."""
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in megabytes
        """
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_increase(self) -> float:
        """
        Get memory increase from baseline.

        Returns:
            Memory increase in megabytes
        """
        return self.get_memory_usage() - self.baseline_memory

    def check_memory_leak(self, threshold_mb: float = 100.0) -> bool:
        """
        Check for potential memory leak.

        Args:
            threshold_mb: Threshold for warning in MB

        Returns:
            True if potential leak detected
        """
        increase = self.get_memory_increase()
        return increase > threshold_mb

    @staticmethod
    def optimize_numpy_array(arr: np.ndarray) -> np.ndarray:
        """
        Optimize numpy array memory usage.

        Args:
            arr: Input array

        Returns:
            Optimized array
        """
        # Use smallest dtype that can hold the data
        if arr.dtype == np.float64:
            if np.all(np.abs(arr) < 3.4e38):
                arr = arr.astype(np.float32)

        elif arr.dtype == np.int64:
            max_val = np.max(np.abs(arr))
            if max_val < 32767:
                arr = arr.astype(np.int16)
            elif max_val < 2147483647:
                arr = arr.astype(np.int32)

        return arr


class ResourceManager:
    """
    Manage computational resources for optimal performance.
    """

    def __init__(self, max_threads: int = None):
        """
        Initialize resource manager.

        Args:
            max_threads: Maximum number of threads (auto-detect if None)
        """
        if max_threads is None:
            max_threads = psutil.cpu_count(logical=False) or 4

        self.max_threads = max_threads
        self.active_threads = 0
        self._lock = threading.Lock()

    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.

        Returns:
            CPU usage (0-100)
        """
        return psutil.cpu_percent(interval=0.1)

    def get_available_threads(self) -> int:
        """
        Get number of available threads.

        Returns:
            Available thread count
        """
        with self._lock:
            return self.max_threads - self.active_threads

    def allocate_thread(self) -> bool:
        """
        Allocate a thread if available.

        Returns:
            True if allocated, False otherwise
        """
        with self._lock:
            if self.active_threads < self.max_threads:
                self.active_threads += 1
                return True
            return False

    def release_thread(self):
        """Release an allocated thread."""
        with self._lock:
            if self.active_threads > 0:
                self.active_threads -= 1

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system resource statistics.

        Returns:
            Dictionary with system stats
        """
        return {
            'cpu_percent': self.get_cpu_usage(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'active_threads': self.active_threads,
            'max_threads': self.max_threads
        }


class CacheManager:
    """
    Manage caching for frequently accessed data.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum cache size
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None
        """
        with self._lock:
            if key in self.cache:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
            return None

    def put(self, key: str, value: Any):
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Evict if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            self.cache[key] = value
            self.access_count[key] = 1

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_count:
            return

        # Find least accessed key
        lru_key = min(self.access_count, key=self.access_count.get)

        # Remove from cache
        del self.cache[lru_key]
        del self.access_count[lru_key]

    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
            }


class AdaptiveThrottler:
    """
    Adaptively throttle processing based on system load.
    """

    def __init__(self, target_cpu_percent: float = 80.0):
        """
        Initialize adaptive throttler.

        Args:
            target_cpu_percent: Target CPU usage percentage
        """
        self.target_cpu_percent = target_cpu_percent
        self.throttle_amount = 0.0  # seconds
        self.adjustment_step = 0.001  # 1ms

    def should_throttle(self) -> bool:
        """
        Check if processing should be throttled.

        Returns:
            True if should throttle
        """
        cpu_usage = psutil.cpu_percent(interval=0.01)

        if cpu_usage > self.target_cpu_percent:
            # Increase throttle
            self.throttle_amount += self.adjustment_step
            return True
        else:
            # Decrease throttle
            self.throttle_amount = max(0, self.throttle_amount - self.adjustment_step)
            return False

    def throttle(self):
        """Apply throttling delay if needed."""
        if self.throttle_amount > 0:
            time.sleep(self.throttle_amount)


class BatchProcessor:
    """
    Process items in batches for improved efficiency.
    """

    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        """
        Initialize batch processor.

        Args:
            batch_size: Maximum batch size
            timeout: Maximum wait time for batch
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.batch: List[Any] = []
        self.last_process_time = time.time()
        self._lock = threading.Lock()

    def add(self, item: Any) -> Optional[List[Any]]:
        """
        Add item to batch.

        Args:
            item: Item to add

        Returns:
            Full batch if ready to process, None otherwise
        """
        with self._lock:
            self.batch.append(item)

            # Check if batch is full
            if len(self.batch) >= self.batch_size:
                return self._flush_batch()

            # Check if timeout exceeded
            if time.time() - self.last_process_time > self.timeout:
                return self._flush_batch()

            return None

    def _flush_batch(self) -> List[Any]:
        """Flush current batch."""
        batch = self.batch
        self.batch = []
        self.last_process_time = time.time()
        return batch

    def flush(self) -> List[Any]:
        """Force flush of current batch."""
        with self._lock:
            return self._flush_batch()


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        import io

        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()

        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        print(s.getvalue())

        return result

    return wrapper


def optimize_for_realtime(priority: str = 'high') -> Callable:
    """
    Decorator to optimize function for real-time processing.

    Args:
        priority: Thread priority ('high', 'normal', 'low')

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set thread priority (platform-specific)
            import sys
            if sys.platform == 'linux':
                try:
                    import os
                    # Set nice value (-20 to 19, lower is higher priority)
                    if priority == 'high':
                        os.nice(-10)
                    elif priority == 'low':
                        os.nice(10)
                except PermissionError:
                    pass  # Requires privileges

            return func(*args, **kwargs)

        return wrapper

    return decorator


class PerformanceOptimizer:
    """
    Comprehensive performance optimizer combining multiple strategies.
    """

    def __init__(self):
        """Initialize performance optimizer."""
        self.latency_optimizer = LatencyOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.resource_manager = ResourceManager()
        self.cache_manager = CacheManager()
        self.throttler = AdaptiveThrottler()

    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization report.

        Returns:
            Dictionary with optimization statistics
        """
        return {
            'latency': {
                'average_ms': self.latency_optimizer.get_average_latency(),
                'meeting_target': self.latency_optimizer.is_meeting_target()
            },
            'memory': {
                'current_mb': self.memory_optimizer.get_memory_usage(),
                'increase_mb': self.memory_optimizer.get_memory_increase(),
                'potential_leak': self.memory_optimizer.check_memory_leak()
            },
            'resources': self.resource_manager.get_system_stats(),
            'cache': self.cache_manager.get_stats(),
            'throttle': {
                'active': self.throttler.throttle_amount > 0,
                'amount_ms': self.throttler.throttle_amount * 1000
            }
        }

    def suggest_optimizations(self) -> List[str]:
        """
        Suggest optimizations based on current performance.

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        report = self.get_optimization_report()

        # Latency suggestions
        if not report['latency']['meeting_target']:
            suggestions.append(
                f"Latency ({report['latency']['average_ms']:.2f}ms) exceeds target. "
                "Consider reducing image resolution or detection frequency."
            )

        # Memory suggestions
        if report['memory']['potential_leak']:
            suggestions.append(
                f"Potential memory leak detected ({report['memory']['increase_mb']:.2f}MB increase). "
                "Check for unbounded buffers or caches."
            )

        # CPU suggestions
        if report['resources']['cpu_percent'] > 90:
            suggestions.append(
                f"High CPU usage ({report['resources']['cpu_percent']:.1f}%). "
                "Consider reducing processing threads or enabling throttling."
            )

        # Cache suggestions
        if report['cache']['utilization'] > 0.9:
            suggestions.append(
                "Cache nearly full. Consider increasing cache size."
            )

        return suggestions
