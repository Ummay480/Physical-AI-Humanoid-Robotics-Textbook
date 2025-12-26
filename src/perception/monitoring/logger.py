"""
Comprehensive logging and monitoring utilities for the perception system.

Provides structured logging, performance monitoring, metrics collection,
and health check capabilities.
"""
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from collections import defaultdict, deque


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthStatus:
    """System health status."""
    component: str
    status: str  # OK, WARNING, ERROR, CRITICAL
    message: str
    timestamp: float
    details: Dict[str, Any] = None


class PerceptionLogger:
    """
    Enhanced logger for the perception system with structured logging.
    """

    def __init__(self, name: str, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize perception logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, msg: str, **kwargs):
        """Log debug message with structured data."""
        self._log(logging.DEBUG, msg, kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message with structured data."""
        self._log(logging.INFO, msg, kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message with structured data."""
        self._log(logging.WARNING, msg, kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message with structured data."""
        self._log(logging.ERROR, msg, kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message with structured data."""
        self._log(logging.CRITICAL, msg, kwargs)

    def _log(self, level: int, msg: str, data: Dict):
        """Internal logging with structured data."""
        if data:
            structured_msg = f"{msg} | {json.dumps(data)}"
        else:
            structured_msg = msg

        self.logger.log(level, structured_msg)

    def log_performance(self, operation: str, duration: float, **metadata):
        """Log performance metrics."""
        self.info(
            f"Performance: {operation}",
            duration_ms=duration * 1000,
            operation=operation,
            **metadata
        )


class PerformanceMonitor:
    """
    Monitor and track performance metrics.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()

        # Start time
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, unit: str = "",
                     metadata: Optional[Dict] = None):
        """
        Record a performance metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        with self._lock:
            self.metrics[name].append(metric)

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if metric_name not in self.metrics:
                return {}

            values = [m.value for m in self.metrics[metric_name]]

            if not values:
                return {}

            import numpy as np
            return {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        with self._lock:
            return {
                name: self.get_statistics(name)
                for name in self.metrics.keys()
            }

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time

    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()


class HealthMonitor:
    """
    Monitor system health and component status.
    """

    def __init__(self):
        """Initialize health monitor."""
        self.health_status: Dict[str, HealthStatus] = {}
        self._lock = threading.Lock()

        # Health check thresholds
        self.thresholds = {
            'latency_ms': 20.0,
            'error_rate': 0.01,  # 1%
            'uptime_min': 0.99    # 99%
        }

    def update_health(self, component: str, status: str, message: str,
                     details: Optional[Dict] = None):
        """
        Update health status for a component.

        Args:
            component: Component name
            status: Status (OK, WARNING, ERROR, CRITICAL)
            message: Status message
            details: Additional details
        """
        health = HealthStatus(
            component=component,
            status=status,
            message=message,
            timestamp=time.time(),
            details=details or {}
        )

        with self._lock:
            self.health_status[component] = health

    def get_health(self, component: Optional[str] = None) -> Dict:
        """
        Get health status.

        Args:
            component: Specific component or None for all

        Returns:
            Health status dictionary
        """
        with self._lock:
            if component:
                health = self.health_status.get(component)
                return asdict(health) if health else {}
            else:
                return {
                    comp: asdict(health)
                    for comp, health in self.health_status.items()
                }

    def get_overall_health(self) -> str:
        """
        Get overall system health status.

        Returns:
            Overall status (OK, WARNING, ERROR, CRITICAL)
        """
        with self._lock:
            if not self.health_status:
                return "UNKNOWN"

            statuses = [h.status for h in self.health_status.values()]

            if "CRITICAL" in statuses:
                return "CRITICAL"
            elif "ERROR" in statuses:
                return "ERROR"
            elif "WARNING" in statuses:
                return "WARNING"
            else:
                return "OK"

    def check_health(self, performance_monitor: PerformanceMonitor) -> Dict:
        """
        Perform comprehensive health check.

        Args:
            performance_monitor: Performance monitor instance

        Returns:
            Health check results
        """
        results = {}

        # Check latency
        latency_stats = performance_monitor.get_statistics('processing_latency')
        if latency_stats:
            avg_latency = latency_stats.get('mean', 0)
            if avg_latency > self.thresholds['latency_ms']:
                self.update_health(
                    'latency', 'WARNING',
                    f"High latency: {avg_latency:.2f}ms",
                    latency_stats
                )
            else:
                self.update_health(
                    'latency', 'OK',
                    f"Latency normal: {avg_latency:.2f}ms",
                    latency_stats
                )

        # Check uptime
        uptime = performance_monitor.get_uptime()
        self.update_health(
            'uptime', 'OK',
            f"Uptime: {uptime:.1f}s"
        )

        # Get overall health
        results['overall'] = self.get_overall_health()
        results['components'] = self.get_health()
        results['timestamp'] = time.time()

        return results


class MetricsCollector:
    """
    Collect and export metrics for monitoring systems.
    """

    def __init__(self, output_dir: str = "metrics"):
        """
        Initialize metrics collector.

        Args:
            output_dir: Directory for metrics output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.metrics: List[PerformanceMetric] = []
        self._lock = threading.Lock()

    def collect(self, metric: PerformanceMetric):
        """Collect a metric."""
        with self._lock:
            self.metrics.append(metric)

    def export_json(self, filename: str = None):
        """
        Export metrics to JSON file.

        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        output_file = self.output_dir / filename

        with self._lock:
            data = {
                'timestamp': datetime.now().isoformat(),
                'count': len(self.metrics),
                'metrics': [m.to_dict() for m in self.metrics]
            }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filename: str = None):
        """
        Export metrics to CSV file.

        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"

        output_file = self.output_dir / filename

        with self._lock:
            if not self.metrics:
                return

            with open(output_file, 'w') as f:
                # Header
                f.write("name,value,unit,timestamp\n")

                # Data
                for metric in self.metrics:
                    f.write(f"{metric.name},{metric.value},{metric.unit},{metric.timestamp}\n")

    def clear(self):
        """Clear collected metrics."""
        with self._lock:
            self.metrics.clear()


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, name: str, monitor: PerformanceMonitor,
                 logger: Optional[PerceptionLogger] = None):
        """
        Initialize timing context.

        Args:
            name: Operation name
            monitor: Performance monitor
            logger: Optional logger
        """
        self.name = name
        self.monitor = monitor
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        duration = time.perf_counter() - self.start_time

        # Record metric
        self.monitor.record_metric(
            self.name,
            duration * 1000,  # Convert to milliseconds
            unit='ms'
        )

        # Log if logger provided
        if self.logger:
            self.logger.log_performance(self.name, duration)

        return False


# Global instances
_performance_monitor = None
_health_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
