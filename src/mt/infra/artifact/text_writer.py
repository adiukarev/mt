import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def write_text(path: str | Path, content: str) -> None:
	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	target.write_text(content, encoding="utf-8")


def write_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
	write_text(path, render_json(payload))


def write_yaml(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
	write_text(path, render_yaml(payload))


def write_markdown(path: str | Path, lines: list[str]) -> None:
	write_text(path, "\n".join(lines).strip() + "\n")


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(target, index=False)


def render_json(payload: Any, ensure_ascii: bool = False, indent: int | None = 2) -> str:
	return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)


def render_yaml(payload: Any, allow_unicode: bool = False) -> str:
	return yaml.safe_dump(_to_yaml_safe(payload), sort_keys=False, allow_unicode=allow_unicode)


def _to_yaml_safe(value: Any) -> Any:
	if isinstance(value, Enum):
		return value.value
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, pd.Timestamp):
		return value.isoformat()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, dict):
		return {str(key): _to_yaml_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_to_yaml_safe(item) for item in value]
	if isinstance(value, set):
		return [_to_yaml_safe(item) for item in sorted(value, key=str)]
	return value


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	with target.open("a", encoding="utf-8") as file_obj:
		file_obj.write(render_json(payload, ensure_ascii=True, indent=None) + "\n")
