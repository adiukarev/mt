import logging
import os
import sys
from pathlib import Path
from typing import Any

import structlog

from mt.infra.observability.logger.filters import ThirdPartyNoiseFilter, configure_third_party_loggers
from mt.infra.observability.logger.renderers import (
	add_observability_fields,
	airflow_renderer,
	file_renderer,
	local_renderer,
)

APP_LOGGER_NAME = "mt"


def configure_runtime_logging(runtime_log_path: str | Path | None = None) -> None:
	execution_mode = os.getenv("MT_EXECUTION_MODE", "local")
	handlers: list[logging.Handler] = [build_stream_handler(execution_mode)]
	if runtime_log_path is not None:
		log_path = Path(runtime_log_path)
		log_path.parent.mkdir(parents=True, exist_ok=True)
		handlers.append(build_file_handler(log_path))

	logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
	configure_third_party_loggers()
	configure_structlog()


def log_info(event: str, *, scope: str, **fields: Any) -> None:
	log("info", event, scope=scope, **fields)


def log_debug(event: str, *, scope: str, **fields: Any) -> None:
	log("debug", event, scope=scope, **fields)


def log_warning(event: str, *, scope: str, **fields: Any) -> None:
	log("warning", event, scope=scope, **fields)


def log(level: str, event: str, *, scope: str, **fields: Any) -> None:
	logger = structlog.stdlib.get_logger(APP_LOGGER_NAME)
	getattr(logger, level)(event, scope=scope, **fields)


def configure_structlog() -> None:
	shared_processors = [
		structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", key="timestamp"),
		structlog.stdlib.add_log_level,
		structlog.stdlib.PositionalArgumentsFormatter(),
		add_observability_fields,
	]
	structlog.configure(
		processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
		wrapper_class=structlog.stdlib.BoundLogger,
		logger_factory=structlog.stdlib.LoggerFactory(),
		cache_logger_on_first_use=True,
	)


def build_stream_handler(execution_mode: str) -> logging.Handler:
	stream = sys.stdout if execution_mode == "airflow" else None
	handler = logging.StreamHandler(stream)
	handler.addFilter(ThirdPartyNoiseFilter())
	handler.setFormatter(
		structlog.stdlib.ProcessorFormatter(
			processor=local_renderer if execution_mode == "local" else airflow_renderer,
			foreign_pre_chain=[
				structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", key="timestamp"),
				structlog.stdlib.add_log_level,
				add_observability_fields,
			],
		)
	)
	return handler


def build_file_handler(log_path: Path) -> logging.Handler:
	handler = logging.FileHandler(log_path, encoding="utf-8")
	handler.addFilter(ThirdPartyNoiseFilter())
	handler.setFormatter(
		structlog.stdlib.ProcessorFormatter(
			processor=file_renderer,
			foreign_pre_chain=[
				structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", key="timestamp"),
				structlog.stdlib.add_log_level,
				add_observability_fields,
			],
		)
	)
	return handler
