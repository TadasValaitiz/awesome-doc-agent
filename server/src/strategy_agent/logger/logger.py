from dataclasses import dataclass
import json
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import UTC, datetime
import abc
import uuid
import os


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
        level: int = logging.DEBUG,
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

    def _safe_serialize(self, obj):
        """Safely serialize objects to JSON, handling non-serializable items."""
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError, ValueError):
            result = {}
            for key, value in obj.items():
                try:
                    # Try to serialize individual value
                    json.dumps({key: value})
                    result[key] = value
                except (TypeError, OverflowError, ValueError):
                    # If not serializable, store string representation
                    result[key] = f"<non-serializable: {type(value).__name__}>"
            return json.dumps(result)

    def log(self, message: LogMessage) -> None:
        """Log message to console with formatting."""
        if logging.getLevelName(message.level) < self.level:
            return

        # Skip if message tags don't match filter tags
        if self.tags and not (set(message.tags or []) & self.tags):
            return

        formatted_message = message.message
        if message.data:
            formatted_message = (
                f"{formatted_message} - {self._safe_serialize(message.data)}"
            )
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
        log_file_path: str = "logs/logs.log",
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

    def _safe_serialize(self, obj):
        """Safely serialize objects to JSON, handling non-serializable items."""
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError, ValueError):
            result = {}
            for key, value in obj.items():
                try:
                    # Try to serialize individual value
                    json.dumps({key: value})
                    result[key] = value
                except (TypeError, OverflowError, ValueError):
                    # If not serializable, store string representation
                    result[key] = f"<non-serializable: {type(value).__name__}>"
            return json.dumps(result)

    def log(self, message: LogMessage) -> None:
        """Log message to file with formatting."""
        if logging.getLevelName(message.level) < self.level:
            return

        # Skip if message tags don't match filter tags
        if self.tags and not (set(message.tags or []) & self.tags):
            return

        formatted_message = message.message
        if message.data:
            formatted_message = (
                f"{formatted_message} - {self._safe_serialize(message.data)}"
            )
        if message.stacktrace:
            formatted_message = (
                f"{formatted_message}\nStacktrace:\n{message.stacktrace}"
            )
        if message.tags:
            formatted_message = f"{formatted_message} [tags: {', '.join(message.tags)}]"

        log_func = getattr(self.logger, message.level.lower())
        log_func(formatted_message)


class TradingLogger:
    """Multi-output logger for trading application using composition pattern."""

    def __init__(self, session_id: str, loggers: List[BaseLogger]):
        self.loggers = loggers
        self.session_id = session_id

    def _get_stacktrace(self) -> Optional[str]:
        """Get formatted stacktrace for error messages."""
        try:
            return traceback.format_exc()
        except Exception:
            return None

    def _create_log_message(
        self,
        level: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        include_stacktrace: bool = False,
    ) -> LogMessage:
        """Create a structured log message."""
        stacktrace = self._get_stacktrace() if include_stacktrace else None
        return LogMessage(
            session_id=self.session_id,
            timestamp=datetime.now(UTC).isoformat(),
            level=level,
            message=message,
            data=data,
            tags=tags or [],
            stacktrace=stacktrace,
        )

    def _log(
        self,
        level: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        include_stacktrace: bool = False,
    ) -> None:
        """Internal method to handle logging to all configured outputs."""
        log_message = self._create_log_message(
            level, message, data, tags, include_stacktrace
        )

        # Send log message to all configured loggers
        for logger in self.loggers:
            logger.log(log_message)

    def debug(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log debug message."""
        self._log("DEBUG", message, data, tags)

    def info(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log info message."""
        self._log("INFO", message, data, tags)

    def warning(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log warning message."""
        self._log("WARNING", message, data, tags)

    def error(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        exception: Optional[Exception] = None,
        include_stacktrace: bool = True,
    ) -> None:
        """Log error message with optional stacktrace."""

        error_data = {
            "error_type": type(exception).__name__ if exception else None,
            "error": str(exception) if exception else None,
            "stacktrace": traceback.format_exc() if exception else None,
        }
        data = {**(data or {}), **error_data}

        error_tags = list(set(["error"] + (tags or [])))

        self._log(
            "ERROR", message, data, error_tags, include_stacktrace=include_stacktrace
        )

    def critical(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        include_stacktrace: bool = True,
    ) -> None:
        """Log critical message with optional stacktrace."""
        self._log("CRITICAL", message, data, tags, include_stacktrace)

    def trade(
        self,
        action: str,
        symbol: str,
        price: float,
        size: float,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log trade-specific message."""
        trade_data = {
            "action": action,
            "symbol": symbol,
            "price": price,
            "size": size,
            **(data or {}),
        }
        trade_tags = ["trade", action.lower(), symbol.lower(), *(tags or [])]
        self.info(f"Trade: {action} {symbol}", trade_data, trade_tags)

    def metric(
        self,
        name: str,
        value: Union[int, float],
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log metric value."""
        metric_data = {
            "metric": name,
            "value": value,
            **(data or {}),
        }
        metric_tags = ["metric", name.lower(), *(tags or [])]
        self.info(f"Metric: {name}={value}", metric_data, metric_tags)


class ServerLogger:
    """Multi-output logger for server application using composition pattern."""

    def __init__(self, session_id: str, loggers: List[BaseLogger]):
        self.loggers = loggers
        self.session_id = session_id

    def _get_stacktrace(self) -> Optional[str]:
        """Get formatted stacktrace for error messages."""
        try:
            return traceback.format_exc()
        except Exception:
            return None

    def _create_log_message(
        self,
        level: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        include_stacktrace: bool = False,
    ) -> LogMessage:
        """Create a structured log message."""
        stacktrace = self._get_stacktrace() if include_stacktrace else None
        return LogMessage(
            session_id=self.session_id,
            timestamp=datetime.now(UTC).isoformat(),
            level=level,
            message=message,
            data=data,
            tags=tags or [],
            stacktrace=stacktrace,
        )

    def _log(
        self,
        level: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        include_stacktrace: bool = False,
    ) -> None:
        """Internal method to handle logging to all configured outputs."""
        log_message = self._create_log_message(
            level, message, data, tags, include_stacktrace
        )

        # Send log message to all configured loggers
        for logger in self.loggers:
            logger.log(log_message)

    def debug(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log debug message."""
        server_tags = ["server"] + (tags or [])
        self._log("DEBUG", message, data, server_tags)

    def info(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log info message."""
        server_tags = ["server"] + (tags or [])
        self._log("INFO", message, data, server_tags)

    def warning(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log warning message."""
        server_tags = ["server"] + (tags or [])
        self._log("WARNING", message, data, server_tags)

    def error(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        exception: Optional[Exception] = None,
        include_stacktrace: bool = True,
    ) -> None:
        """Log error message with optional stacktrace."""
        error_data = {
            "error_type": type(exception).__name__ if exception else None,
            "error": str(exception) if exception else None,
            "stacktrace": traceback.format_exc() if exception else None,
        }
        data = {**(data or {}), **error_data}

        error_tags = list(set(["server", "error"] + (tags or [])))

        self._log(
            "ERROR", message, data, error_tags, include_stacktrace=include_stacktrace
        )

    def critical(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        include_stacktrace: bool = True,
    ) -> None:
        """Log critical message with optional stacktrace."""
        server_tags = ["server"] + (tags or [])
        self._log("CRITICAL", message, data, server_tags, include_stacktrace)

    def request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_ms: float,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log HTTP request-specific message."""
        request_data = {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **(data or {}),
        }
        request_tags = ["server", "request", method.lower(), *(tags or [])]
        self.info(
            f"Request: {method} {endpoint} {status_code}", request_data, request_tags
        )

    def startup(
        self,
        port: int,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log server startup information."""
        startup_data = {
            "port": port,
            **(data or {}),
        }
        startup_tags = ["server", "startup", *(tags or [])]
        self.info(f"Server started on port {port}", startup_data, startup_tags)


logger_session = str(uuid.uuid4())
server_logger = ServerLogger(
    session_id=logger_session,
    loggers=[
        ConsoleLogger(session_id=logger_session, level=logging.DEBUG),
        FileLogger(
            session_id=logger_session,
            log_file_path="logs/server.log",
            level=logging.DEBUG,
        ),
    ],
)
