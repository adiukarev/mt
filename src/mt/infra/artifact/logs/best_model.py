import logging
from pathlib import Path
from typing import Any
from mt.domain.model import ModelResult


def log_best_model_save(model_name: str, horizons: list[str], root: Path) -> None:
	logging.info(
		f"Финальная модель сохранена | "
		f"model={model_name} | "
		f"horizons={",".join(str(horizon) for horizon in horizons)} | "
		f"path={root}",
	)


def log_best_model_prediction(model_name: str, result: Any, output_path: Path) -> ModelResult:
	logging.info(
		f"Прогноз сохранен | "
		f"model={model_name} | "
		f"rows={result} | "
		f"path={output_path}",
	)
