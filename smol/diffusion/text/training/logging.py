import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER_NAME = "smol.diffusion.text"
_CONFIGURED_HANDLER: logging.Handler | None = None
_BASE_CONTEXT: dict[str, Any] = {}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        context = getattr(record, "context", None)
        if isinstance(context, dict) and context:
            payload.update(context)

        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict) and event_data:
            payload.update(event_data)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class StructuredLogger:
    def __init__(
        self, logger: logging.Logger, context: dict[str, Any] | None = None
    ) -> None:
        self._logger = logger
        self._context = context or {}

    def bind(self, **context: Any) -> "StructuredLogger":
        return StructuredLogger(self._logger, {**self._context, **context})

    def debug(self, message: str, **event_data: Any) -> None:
        self._log(logging.DEBUG, message, **event_data)

    def info(self, message: str, **event_data: Any) -> None:
        self._log(logging.INFO, message, **event_data)

    def warning(self, message: str, **event_data: Any) -> None:
        self._log(logging.WARNING, message, **event_data)

    def error(self, message: str, **event_data: Any) -> None:
        self._log(logging.ERROR, message, **event_data)

    def exception(self, message: str, **event_data: Any) -> None:
        self._logger.exception(
            message, extra={"context": self._context, "event_data": event_data}
        )

    def _log(self, level: int, message: str, **event_data: Any) -> None:
        self._logger.log(
            level, message, extra={"context": self._context, "event_data": event_data}
        )


def init_logging(
    log_path: Path, *, level: int = logging.INFO, **base_context: Any
) -> StructuredLogger:
    global _CONFIGURED_HANDLER, _BASE_CONTEXT

    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger(_LOGGER_NAME)
    root_logger.setLevel(level)
    root_logger.propagate = False

    if _CONFIGURED_HANDLER is not None:
        root_logger.removeHandler(_CONFIGURED_HANDLER)
        _CONFIGURED_HANDLER.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter())
    root_logger.addHandler(handler)

    _CONFIGURED_HANDLER = handler
    _BASE_CONTEXT = dict(base_context)
    return StructuredLogger(root_logger, dict(_BASE_CONTEXT))


def get_logger(name: str | None = None, **context: Any) -> StructuredLogger:
    logger_name = _LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}"
    return StructuredLogger(
        logging.getLogger(logger_name), {**_BASE_CONTEXT, **context}
    )


def shutdown_logging() -> None:
    global _CONFIGURED_HANDLER, _BASE_CONTEXT

    if _CONFIGURED_HANDLER is None:
        return

    root_logger = logging.getLogger(_LOGGER_NAME)
    root_logger.removeHandler(_CONFIGURED_HANDLER)
    _CONFIGURED_HANDLER.close()
    _CONFIGURED_HANDLER = None
    _BASE_CONTEXT = {}
