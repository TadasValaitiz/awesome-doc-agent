from dataclasses import dataclass
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
import abc
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_serialize(obj):
    """Safely serialize objects to JSON, handling non-serializable items."""
    try:
        return json.dumps(obj)
    except (TypeError, OverflowError, ValueError):
        result = {}
        for key, value in obj.items():
            try:
                # Try to serialize individual value
                json.dumps({key: value}, cls=NumpyJSONEncoder)
                result[key] = value
            except (TypeError, OverflowError, ValueError):
                # If not serializable, store string representation
                result[key] = f"<non-serializable: {type(value).__name__}>"
        return json.dumps(result)


@dataclass
class LogMessage:
    """Structure for log messages.

    Attributes:
        timestamp: ISO format timestamp of the log message
        level: str: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Main log message
        data: Additional structured data for the log entry
        tags: List of tags for categorizing and filtering logs
        stacktrace: Full stacktrace for error messages (only for ERROR and CRITICAL levels)
    """

    session_id: str
    timestamp: str
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    stacktrace: Optional[str] = None


class BaseLogger(abc.ABC):
    """Abstract base class for all loggers."""

    def __init__(self, session_id: str, level: int = logging.INFO):
        self.session_id = session_id
        self.level = level

    @abc.abstractmethod
    def log(self, message: LogMessage) -> None:
        """Log the message using the specific logger implementation."""
        pass


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""

    COLORS = {
        "DEBUG": "\033[36;1m",  # Bright cyan
        "INFO": "\033[0m",  # Default/white
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[31;1m",  # Bold red
        "RESET": "\033[0m",  # Reset to default
    }

    def __init__(self, fmt):
        super().__init__(fmt)

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


class ConsoleLogger(BaseLogger):
    """Logger implementation for console output."""

    def __init__(
        self,
        session_id: str,
        level: int = logging.INFO,
        tags: Optional[List[str]] = None,
    ):
        super().__init__(session_id, level)
        self.logger = logging.getLogger(f"console_logger_{session_id}")
        self.logger.setLevel(level)
        self.tags = set(tags) if tags else None

        # Only add handler if none exist
        if not self.logger.handlers:
            # Setup console handler
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = ColorFormatter("%(asctime)s [%(levelname)s] %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log(self, message: LogMessage) -> None:
        """Log message to console with formatting."""
        if logging.getLevelName(message.level) < self.level:
            return

        # Skip if message tags don't match filter tags
        if self.tags and not (set(message.tags or []) & self.tags):
            return

        formatted_message = message.message
        if message.data:
            formatted_message = f"{formatted_message} - {_safe_serialize(message.data)}"
        if message.stacktrace:
            formatted_message = (
                f"{formatted_message}\nStacktrace:\n{message.stacktrace}"
            )
        if message.tags:
            formatted_message = f"{formatted_message} [tags: {', '.join(message.tags)}]"

        log_func = getattr(self.logger, message.level.lower())
        log_func(formatted_message)


class FileLogger(BaseLogger):
    """Logger implementation for file output."""

    def __init__(
        self,
        session_id: str,
        level: int = logging.DEBUG,
        tags: Optional[List[str]] = None,
        log_file_path: str = "data/optimization/logs.log",
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5,
    ):
        super().__init__(session_id, level)
        self.logger = logging.getLogger(f"file_logger_{session_id}")
        self.logger.setLevel(level)
        self.tags = set(tags) if tags else None
        self.log_file_path = log_file_path

        # Only add handler if none exist
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            # Setup file handler with rotation
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=max_bytes, backupCount=backup_count
            )
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, message: LogMessage) -> None:
        """Log message to file with formatting."""
        if logging.getLevelName(message.level) < self.level:
            return

        # Skip if message tags don't match filter tags
        if self.tags and not (set(message.tags or []) & self.tags):
            return

        formatted_message = message.message
        if message.data:
            formatted_message = f"{formatted_message} - {_safe_serialize(message.data)}"
        if message.stacktrace:
            formatted_message = (
                f"{formatted_message}\nStacktrace:\n{message.stacktrace}"
            )
        if message.tags:
            formatted_message = f"{formatted_message} [tags: {', '.join(message.tags)}]"

        log_func = getattr(self.logger, message.level.lower())
        log_func(formatted_message)
