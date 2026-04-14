import json
from pathlib import Path
from typing import Any

import yaml


def dump_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
	"""Сохранить JSON в стабильном формате"""

	Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
	"""Сохранить YAML для конфигурационных артефактов"""

	Path(path).write_text(
		yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
		encoding="utf-8",
	)
