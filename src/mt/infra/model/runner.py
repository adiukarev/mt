import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from mt.domain.manifest import DLManifest
from mt.domain.model import ForecastModelAdapter, ModelInfo, ModelResult
from mt.infra.model.registry import build_model_adapter
from mt.infra.artifact.logs.model import (
	log_model_fit_and_predict_horizon_start,
	log_model_fit_and_predict_horizon_end
)

SUPPRESSED_CONVERGENCE_WARNING_MODELS: set[str] = set()


@dataclass(slots=True)
class WindowExecutionResult:
	"""Результат прогона модели по набору окон"""

	rows: list[dict[str, object]]
	warnings: list[str]
	train_time_seconds: float
	inference_time_seconds: float
	suppressed_convergence_warnings: int = 0


@dataclass(slots=True)
class WindowFrames:
	"""Подготовленные фреймы для одного окна и горизонта"""

	train_frame: pd.DataFrame
	valid_predict: pd.DataFrame
	target_column: str


def run_model(
	model_name: str,
	supervised: pd.DataFrame,
	feature_columns: list[str],
	windows: pd.DataFrame,
	seed: int,
	model_params: dict[str, object] | None = None,
	dl_manifest: DLManifest | None = None,
) -> ModelResult:
	"""Прогнать одну модель по всем rolling-окнам backtesting"""

	model_wall_start = time.perf_counter()
	adapter = build_model_adapter(model_name, model_params=model_params, dl_manifest=dl_manifest)
	prepared_frame = adapter.prepare_frame(supervised)
	resolved_feature_columns = adapter.resolve_feature_columns(prepared_frame, feature_columns)

	result = _run_backtest_windows(
		adapter=adapter,
		model_name=model_name,
		prepared_frame=prepared_frame,
		feature_columns=resolved_feature_columns,
		windows=windows,
		seed=seed,
	)

	predictions = pd.DataFrame(result.rows)
	if not predictions.empty and predictions[["actual", "prediction"]].isna().any().any():
		raise ValueError()

	return ModelResult(
		info=adapter.get_model_info(),
		predictions=predictions,
		warnings=sorted(set(result.warnings)),
		train_time_seconds=result.train_time_seconds,
		inference_time_seconds=result.inference_time_seconds,
		wall_time_seconds=time.perf_counter() - model_wall_start,
		used_feature_columns=resolved_feature_columns,
	)


def _run_backtest_windows(
	adapter: ForecastModelAdapter,
	model_name: str,
	prepared_frame: pd.DataFrame,
	feature_columns: list[str],
	windows: pd.DataFrame,
	seed: int,
) -> WindowExecutionResult:
	"""Выполнить обучение и прогнозирование по всем окнам модели"""

	rows: list[dict[str, object]] = []
	warnings: list[str] = []
	train_time_total = 0.0
	inference_time_total = 0.0
	suppressed_convergence_warnings = 0
	model_info = adapter.get_model_info()

	for horizon, horizon_windows in _iter_horizon_windows(windows):
		log_model_fit_and_predict_horizon_start(
			model_name,
			horizon,
			horizon_windows
		)

		for window in horizon_windows.itertuples(index=False):
			window_idx = len(rows)
			window_frames = _prepare_window_frames(
				adapter=adapter,
				prepared_frame=prepared_frame,
				feature_columns=feature_columns,
				horizon=horizon,
				forecast_origin=window.forecast_origin,
				train_end=window.train_end,
			)
			if window_frames is None:
				raise ValueError()

			predictions, train_elapsed, inference_elapsed, suppressed_warnings = _fit_and_predict_window(
				adapter=adapter,
				model_name=model_name,
				window_frames=window_frames,
				feature_columns=feature_columns,
				horizon=horizon,
				seed=seed,
			)
			train_time_total += train_elapsed
			inference_time_total += inference_elapsed
			suppressed_convergence_warnings += suppressed_warnings

			if len(predictions) != len(window_frames.valid_predict):
				raise ValueError()

			rows.extend(
				_build_prediction_rows(
					model_info=model_info,
					valid_predict=window_frames.valid_predict,
					target_column=window_frames.target_column,
					forecast_origin=window.forecast_origin,
					target_date=window.test_start,
					horizon=horizon,
					predictions=predictions,
				)
			)

			log_model_fit_and_predict_horizon_end(
				model_name,
				horizon,
				window.forecast_origin,
				len(rows) - window_idx,
			)

	return WindowExecutionResult(
		rows=rows,
		warnings=warnings,
		train_time_seconds=train_time_total,
		inference_time_seconds=inference_time_total,
		suppressed_convergence_warnings=suppressed_convergence_warnings,
	)


def _iter_horizon_windows(windows: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
	"""Сгруппировать окна по горизонту прогноза"""

	return [
		(int(horizon), windows[windows["horizon"] == horizon])
		for horizon in sorted(windows["horizon"].unique())
	]


def _prepare_window_frames(
	adapter: ForecastModelAdapter,
	prepared_frame: pd.DataFrame,
	feature_columns: list[str],
	horizon: int,
	forecast_origin: pd.Timestamp,
	train_end: pd.Timestamp,
) -> WindowFrames | None:
	"""Подготовить train и predict-части для одного окна"""

	target_column = f"target_h{horizon}"
	train_frame = prepared_frame[prepared_frame["week_start"] <= train_end]
	predict_candidates = prepared_frame[prepared_frame["week_start"] == forecast_origin]
	valid_predict = predict_candidates.dropna(subset=[target_column]).copy()
	valid_predict = adapter.select_predict_frame(valid_predict, feature_columns, target_column)
	if valid_predict.empty:
		return None

	return WindowFrames(
		train_frame=train_frame,
		valid_predict=valid_predict,
		target_column=target_column,
	)


def _fit_and_predict_window(
	adapter: ForecastModelAdapter,
	model_name: str,
	window_frames: WindowFrames,
	feature_columns: list[str],
	horizon: int,
	seed: int,
) -> tuple[np.ndarray, float, float, int]:
	"""Обучить модель на окне и сразу получить прогноз"""

	train_start = time.perf_counter()
	suppressed_warnings = _fit_with_warning_suppression(
		adapter=adapter,
		model_name=model_name,
		train_frame=window_frames.train_frame,
		feature_columns=feature_columns,
		target_column=window_frames.target_column,
		horizon=horizon,
		seed=seed,
	)
	train_elapsed = time.perf_counter() - train_start

	predict_start = time.perf_counter()
	predictions = adapter.predict(
		window_frames.valid_predict,
		feature_columns,
		window_frames.target_column,
		horizon,
	)
	inference_elapsed = time.perf_counter() - predict_start

	return predictions, train_elapsed, inference_elapsed, suppressed_warnings


def _fit_with_warning_suppression(
	adapter: ForecastModelAdapter,
	model_name: str,
	train_frame: pd.DataFrame,
	feature_columns: list[str],
	target_column: str,
	horizon: int,
	seed: int,
) -> int:
	if model_name not in SUPPRESSED_CONVERGENCE_WARNING_MODELS:
		adapter.fit(train_frame, feature_columns, target_column, horizon, seed)
		return 0

	with warnings.catch_warnings(record=True) as caught_warnings:
		warnings.simplefilter("always", ConvergenceWarning)
		adapter.fit(train_frame, feature_columns, target_column, horizon, seed)

	return sum(
		1
		for warning_record in caught_warnings
		if issubclass(warning_record.category, ConvergenceWarning)
	)


def _build_prediction_rows(
	model_info: ModelInfo,
	valid_predict: pd.DataFrame,
	target_column: str,
	forecast_origin: pd.Timestamp,
	target_date: pd.Timestamp,
	horizon: int,
	predictions: np.ndarray,
) -> list[dict[str, object]]:
	"""Преобразовать прогнозы окна в строки общего датафрейма"""

	rows: list[dict[str, object]] = []
	for (_, test_row), prediction in zip(valid_predict.iterrows(), predictions, strict=False):
		rows.append(
			{
				"model_name": model_info.model_name,
				"model_family": model_info.model_family,
				"series_id": test_row["series_id"],
				"category": test_row["category"],
				"segment_label": test_row.get("segment_label"),
				"forecast_origin": forecast_origin,
				"target_date": target_date,
				"horizon": horizon,
				"actual": test_row[target_column],
				"prediction": float(prediction),
			}
		)
	return rows
