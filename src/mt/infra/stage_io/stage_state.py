from pathlib import Path
from typing import Any

from mt.infra.artifact.text_writer import write_json


def save_stage_state(
	stage_states_dir: str | Path,
	index: int,
	stage_name: str,
	payload: dict[str, Any],
) -> Path:
	path = Path(stage_states_dir) / f"{index:03d}_{stage_name}.json"
	write_json(path, payload)
	return path
