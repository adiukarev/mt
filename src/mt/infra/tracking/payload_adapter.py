from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_mapping(value: Any) -> dict[str, Any]:
	if hasattr(value, "to_dict"):
		return value.to_dict()
	if is_dataclass(value):
		return asdict(value)
	if isinstance(value, dict):
		return value
	return {}


def flatten_mapping(
	target: dict[str, Any],
	payload: dict[str, Any],
	prefix: str,
	skip_keys: set[str] | None = None,
) -> None:
	for key, value in payload.items():
		if skip_keys and key in skip_keys:
			continue
		full_key = f"{prefix}.{key}"
		_flatten_value(target, full_key, value)


def _flatten_value(target: dict[str, Any], key: str, value: Any) -> None:
	if isinstance(value, dict):
		flatten_mapping(target, value, key, skip_keys=None)
		return
	if isinstance(value, (list, tuple)):
		items = list(value)
		if not items:
			return
		if all(isinstance(item, (str, int, float, bool)) for item in items):
			add_scalar_param(target, key, items)
			return
		for index, item in enumerate(items):
			_flatten_value(target, f"{key}.{index}", item)
		return
	add_scalar_param(target, key, value)


def add_scalar_param(target: dict[str, Any], key: str, value: Any) -> None:
	if value is None:
		return
	if isinstance(value, (str, int, float, bool)):
		target[key] = value
		return
	if isinstance(value, Path):
		target[key] = str(value)
		return
	if isinstance(value, (list, tuple, set)):
		items = list(value)
		if items and all(isinstance(item, (str, int, float, bool)) for item in items):
			target[key] = ", ".join(str(item) for item in items)


def add_scalar_metric(target: dict[str, float], key: str, value: Any) -> None:
	if value is None or isinstance(value, str):
		return
	try:
		target[key] = float(value)
	except (TypeError, ValueError):
		return


def stringify_param(value: Any) -> str | int | float:
	if isinstance(value, (bool, int, float, str)):
		return value
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, (list, tuple, set)):
		return ", ".join(str(item) for item in value)
	return str(value)
