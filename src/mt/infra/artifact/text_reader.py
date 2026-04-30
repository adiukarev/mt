import json
from pathlib import Path
from typing import Any

import yaml


def read_text(path: str | Path) -> str:
	return Path(path).read_text(encoding="utf-8")


def read_json(path: str | Path) -> Any:
	return json.loads(read_text(path))


def read_yaml(path: str | Path) -> Any:
	return yaml.safe_load(read_text(path))


def read_yaml_mapping(path: str | Path) -> dict[str, Any]:
	payload = read_yaml(path) or {}
	if not isinstance(payload, dict):
		raise TypeError(f"Expected mapping in {path}")
	return payload
