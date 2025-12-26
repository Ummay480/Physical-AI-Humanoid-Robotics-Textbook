"""
Utility functions for skills and subagents framework.

Provides helper functions for UUID generation, timestamps, formatting,
and common operations.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from uuid import UUID, uuid4
import json
import hashlib


# UUID utilities

def generate_agent_id() -> UUID:
    """
    Generate a unique identifier for a subagent.

    Returns:
        UUID: New UUID4 identifier
    """
    return uuid4()


def generate_message_id() -> UUID:
    """
    Generate a unique identifier for a message.

    Returns:
        UUID: New UUID4 identifier
    """
    return uuid4()


def is_valid_uuid(value: str) -> bool:
    """
    Check if a string is a valid UUID.

    Args:
        value: String to validate

    Returns:
        True if valid UUID, False otherwise
    """
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


# Timestamp utilities

def get_current_timestamp() -> datetime:
    """
    Get current UTC timestamp.

    Returns:
        datetime: Current datetime in UTC
    """
    return datetime.utcnow()


def get_current_timestamp_iso() -> str:
    """
    Get current UTC timestamp as ISO 8601 string.

    Returns:
        str: ISO 8601 formatted timestamp
    """
    return datetime.utcnow().isoformat()


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime as ISO 8601 string.

    Args:
        dt: Datetime to format

    Returns:
        str: ISO 8601 formatted string
    """
    return dt.isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO 8601 timestamp string to datetime.

    Args:
        timestamp_str: ISO 8601 formatted string

    Returns:
        datetime: Parsed datetime object

    Raises:
        ValueError: If timestamp format is invalid
    """
    try:
        return datetime.fromisoformat(timestamp_str)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e


def calculate_duration_seconds(start: datetime, end: datetime) -> float:
    """
    Calculate duration between two timestamps in seconds.

    Args:
        start: Start timestamp
        end: End timestamp

    Returns:
        float: Duration in seconds
    """
    return (end - start).total_seconds()


# JSON utilities

def to_json_string(data: Any, pretty: bool = False) -> str:
    """
    Convert data to JSON string.

    Args:
        data: Data to serialize
        pretty: If True, format with indentation

    Returns:
        str: JSON string
    """
    if pretty:
        return json.dumps(data, indent=2, default=str)
    return json.dumps(data, default=str)


def from_json_string(json_str: str) -> Any:
    """
    Parse JSON string to Python object.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed Python object

    Raises:
        ValueError: If JSON is invalid
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary with default.

    Args:
        dictionary: Dictionary to query
        key: Key to look up
        default: Default value if key not found

    Returns:
        Value from dictionary or default
    """
    return dictionary.get(key, default)


def deep_get(dictionary: Dict[str, Any], path: str, default: Any = None, separator: str = ".") -> Any:
    """
    Get nested value from dictionary using dot notation.

    Args:
        dictionary: Dictionary to query
        path: Dot-separated path (e.g., "user.address.city")
        default: Default value if path not found
        separator: Path separator (default: ".")

    Returns:
        Value at path or default

    Example:
        >>> data = {"user": {"name": "Alice", "address": {"city": "NYC"}}}
        >>> deep_get(data, "user.address.city")
        'NYC'
    """
    keys = path.split(separator)
    value = dictionary

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


# String utilities

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length with suffix.

    Args:
        text: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Filename to sanitize
        replacement: Character to replace invalid chars with

    Returns:
        Sanitized filename
    """
    import re

    # Remove invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', replacement, filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")

    return sanitized


# Hash utilities

def compute_hash(data: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of string data.

    Args:
        data: String data to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)

    Returns:
        Hexadecimal hash string

    Raises:
        ValueError: If algorithm is not supported
    """
    algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
    }

    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    hash_func = algorithms[algorithm]
    return hash_func(data.encode("utf-8")).hexdigest()


def compute_dict_hash(data: Dict[str, Any], algorithm: str = "sha256") -> str:
    """
    Compute hash of dictionary by converting to sorted JSON.

    Args:
        data: Dictionary to hash
        algorithm: Hash algorithm

    Returns:
        Hexadecimal hash string
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return compute_hash(json_str, algorithm)


# Collection utilities

def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks

    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists into a single list.

    Args:
        nested_list: List of lists to flatten

    Returns:
        Flattened list

    Example:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
    """
    return [item for sublist in nested_list for item in sublist]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries (later dicts override earlier ones).

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary

    Example:
        >>> merge_dicts({"a": 1}, {"b": 2}, {"a": 3})
        {'a': 3, 'b': 2}
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


# Formatting utilities

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Example:
        >>> format_duration(125.5)
        '2m 5.5s'
        >>> format_duration(3665)
        '1h 1m 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds:.1f}s"
        return f"{minutes}m"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    parts = [f"{hours}h"]
    if remaining_minutes > 0:
        parts.append(f"{remaining_minutes}m")
    if remaining_seconds > 0:
        parts.append(f"{remaining_seconds:.1f}s")

    return " ".join(parts)


def format_bytes(bytes_count: int) -> str:
    """
    Format byte count to human-readable string.

    Args:
        bytes_count: Number of bytes

    Returns:
        Formatted byte string

    Example:
        >>> format_bytes(1536)
        '1.5 KB'
        >>> format_bytes(1048576)
        '1.0 MB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format float as percentage string.

    Args:
        value: Value between 0.0 and 1.0
        decimals: Number of decimal places

    Returns:
        Formatted percentage string

    Example:
        >>> format_percentage(0.7543, 2)
        '75.43%'
    """
    return f"{value * 100:.{decimals}f}%"


# Retry utilities

def exponential_backoff(attempt: int, base_delay: float = 1.0, multiplier: float = 2.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Retry attempt number (0-indexed)
        base_delay: Initial delay in seconds
        multiplier: Exponential multiplier
        max_delay: Maximum delay cap

    Returns:
        Delay in seconds

    Example:
        >>> exponential_backoff(0)  # First retry
        1.0
        >>> exponential_backoff(3)  # Fourth retry
        8.0
    """
    delay = base_delay * (multiplier**attempt)
    return min(delay, max_delay)
