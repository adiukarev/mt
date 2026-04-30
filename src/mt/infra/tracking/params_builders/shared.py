from pathlib import Path
from typing import Any

from mt.infra.tracking.payload_adapter import add_scalar_param


def add_dataset_params(params: dict[str, Any], payload: dict[str, Any]) -> None:
	add_scalar_param(params, "dataset.kind", nested_value(payload, "dataset", "kind"))
	add_scalar_param(
		params,
		"dataset.aggregation_level",
		nested_value(payload, "dataset", "aggregation_level"),
	)
	add_scalar_param(params, "dataset.target_name", nested_value(payload, "dataset", "target_name"))


def add_runtime_params(params: dict[str, Any], payload: dict[str, Any]) -> None:
	add_scalar_param(params, "runtime.seed", nested_value(payload, "runtime", "seed"))


def artifact_basename(payload: dict[str, Any]) -> str | None:
	runtime = payload.get("runtime")
	if not isinstance(runtime, dict):
		return None
	artifacts_dir = runtime.get("artifacts_dir")
	if not isinstance(artifacts_dir, str) or not artifacts_dir.strip():
		return None
	return Path(artifacts_dir).name


def nested_value(payload: dict[str, Any], *keys: str) -> Any:
	current: Any = payload
	for key in keys:
		if not isinstance(current, dict):
			return None
		current = current.get(key)
	return current


def nested_str(payload: dict[str, Any], *keys: str) -> str | None:
	value = nested_value(payload, *keys)
	if value is None:
		return None
	resolved = str(value).strip()
	return resolved or None


def add_tag(tags: dict[str, str], key: str, value: str | None) -> None:
	if value is None:
		return
	tags[key] = value


def render_tag_list(value: Any) -> str | None:
	if not isinstance(value, list) or not value:
		return None
	return ",".join(str(item) for item in value)
