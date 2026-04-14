import logging
from pathlib import Path

import pandas as pd

from mt.domain.model import ModelResult


def log_model_runner_start(model_name: str, windows: pd.DataFrame):
	logging.info(
		f"model {model_name} started | "
		f"windows={len(windows)}"
	)


def log_model_runner_end(result: ModelResult, model_dir: Path) -> None:
	logging.info(
		f"model {result.info.model_name} completed | "
		f"family={result.info.model_family} | "
		f"train_time={result.train_time_seconds or 0.0:.3f}s | "
		f"infer_time={result.inference_time_seconds or 0.0:.3f}s | "
		f"wall_time={result.wall_time_seconds or 0.0:.3f}s | "
		f"artifacts={model_dir}",
	)


def log_model_fit_and_predict_horizon_start(
	model_name: str,
	horizon: int,
	horizon_windows: pd.DataFrame
) -> None:
	logging.info(
		f"model {model_name} started | "
		f"horizon {horizon} | "
		f"windows={len(horizon_windows)}"
	)


def log_model_fit_and_predict_horizon_end(
	model_name: str,
	horizon: int,
	forecast_origin: pd.Timestamp,
	prediction_rows: int,
) -> None:
	logging.info(
		f"model {model_name} window_predicted | "
		f"horizon {horizon} | "
		f"origin {forecast_origin} | "
		f"prediction_rows={prediction_rows}"
	)
