from typing import Any

from structlog.typing import EventDict

from mt.infra.observability.runtime.context_store import get_observability

DEFERRED_KEYS = {"artifacts", "artifacts_dir", "path", "table", "models"}
FIELD_ALIASES = {
	"train_time": "train",
	"infer_time": "infer",
	"wall_time": "wall",
	"artifacts_dir": "artifacts",
	"feature_set": "features",
	"model_name": "model",
}


def add_observability_fields(_: Any, __: str, event_dict: EventDict) -> EventDict:
	observability = get_observability()
	event_dict["run_key"] = observability.run_key if observability is not None else "-"
	event_dict["pipeline_type"] = observability.pipeline_type if observability is not None else "-"
	event_dict["tracking_run_id"] = (
		observability.tracking_run_id
		if observability is not None and observability.tracking_run_id
		else "-"
	)
	return event_dict


def local_renderer(_: Any, __: str, event_dict: EventDict) -> str:
	lines = build_local_lines(event_dict)
	if not lines:
		lines = [f"{format_local_timestamp(event_dict)} {str(event_dict.get('level', 'INFO')).upper()}"]
	return "\n".join(lines)


def airflow_renderer(_: Any, __: str, event_dict: EventDict) -> str:
	parts = [
		str(event_dict.get("timestamp", "-")),
		str(event_dict.get("level", "info")).upper(),
		f"pipeline={event_dict.get('pipeline_type', '-')}",
		f"run_key={event_dict.get('run_key', '-')}",
		f"tracking_run_id={event_dict.get('tracking_run_id', '-')}",
		f"event={event_dict.get('event', '-')}",
	]
	for key, value in iter_renderable_fields(event_dict):
		parts.append(f"{key}={value}")
	return " | ".join(parts)


def file_renderer(_: Any, __: str, event_dict: EventDict) -> str:
	parts = [
		str(event_dict.get("timestamp", "-")),
		str(event_dict.get("level", "info")).upper(),
		f"scope={event_dict.get('scope', '-')}",
		f"event={event_dict.get('event', '-')}",
	]
	for key, value in iter_renderable_fields(event_dict):
		parts.append(f"{key}={value}")
	return " | ".join(parts)


def build_local_lines(event_dict: EventDict) -> list[str]:
	if "scope" not in event_dict:
		return [build_foreign_log_line(event_dict)]
	scope, action = resolve_scope_and_action(event_dict)
	inline_fields: list[str] = []
	deferred_blocks: list[str] = []
	for key, value in iter_renderable_fields(event_dict):
		value_str = str(value)
		if key in DEFERRED_KEYS or "\n" in value_str:
			deferred_blocks.append(f"{FIELD_ALIASES.get(key, key)}:")
			deferred_blocks.extend(normalize_multiline_value(value_str))
			continue
		inline_fields.append(format_inline_field(key, normalize_field_value(key, value_str)))

	lines = [build_head_line(event_dict, scope, action, inline_fields)]
	if deferred_blocks:
		lines.extend(build_deferred_lines(deferred_blocks))
	return lines


def resolve_scope_and_action(event_dict: EventDict) -> tuple[str, str]:
	level = str(event_dict.get("level", "info")).upper()
	scope = str(event_dict.get("scope") or level).upper()
	action = str(event_dict.get("event", ""))
	if level in {"WARNING", "ERROR", "CRITICAL"}:
		prefix = str(event_dict.get("scope") or "log")
		return level, f"{prefix}:{action}".strip(":")
	return scope, action


def iter_renderable_fields(event_dict: EventDict) -> list[tuple[str, Any]]:
	skip_keys = {
		"event",
		"scope",
		"level",
		"timestamp",
		"_record",
		"_from_structlog",
		"run_key",
		"pipeline_type",
		"tracking_run_id",
	}
	return [(key, value) for key, value in event_dict.items() if key not in skip_keys]


def build_head_line(event_dict: EventDict, scope: str, action: str, fields: list[str]) -> str:
	timestamp = format_local_timestamp(event_dict)
	base = f"{timestamp} {scope} {action}"
	if not fields:
		return base
	return f"{base} {' '.join(fields)}"


def build_deferred_lines(lines: list[str]) -> list[str]:
	rendered: list[str] = []
	for line in lines:
		if line.endswith(":"):
			rendered.append(f"  {line}")
			continue
		rendered.append(f"    {line}")
	return rendered


def normalize_multiline_value(value: str) -> list[str]:
	return [line.rstrip() for line in value.splitlines() if line.strip()]


def format_local_timestamp(event_dict: EventDict) -> str:
	value = str(event_dict.get("timestamp", "")).strip()
	if len(value) >= 8:
		return value[-8:]
	return "--:--:--"


def format_inline_field(key: str, value: str) -> str:
	return f"[{FIELD_ALIASES.get(key, key)}={value}]"


def build_foreign_log_line(event_dict: EventDict) -> str:
	timestamp = format_local_timestamp(event_dict)
	level = str(event_dict.get("level", "info")).upper()
	message = str(event_dict.get("event", "")).strip()
	fields = [
		format_inline_field(key, normalize_field_value(key, str(value)))
		for key, value in iter_renderable_fields(event_dict)
	]
	line = f"{timestamp} {level}"
	if message:
		line = f"{line} {message}"
	if fields:
		line = f"{line} {' '.join(fields)}"
	return line


def normalize_field_value(key: str, value: str) -> str:
	if key == "name":
		return shorten_pipeline_name(value)
	return value


def shorten_pipeline_name(value: str) -> str:
	prefixes = ("experiment_", "audit_", "model_", "forecast_", "synthetic_generation_")
	for prefix in prefixes:
		if value.startswith(prefix):
			return value[len(prefix):]
	return value
