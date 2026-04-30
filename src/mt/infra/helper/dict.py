from typing import Any


def get_required_mapping_section(data: dict[str, Any], key: str) -> dict[str, Any]:
	value = data.get(key, {})
	if not isinstance(value, dict):
		raise TypeError()
	return value


def pop_required_mapping_section(data: dict[str, Any], key: str) -> dict[str, Any]:
	value = data.pop(key, {})
	if not isinstance(value, dict):
		raise TypeError()
	return dict(value)
