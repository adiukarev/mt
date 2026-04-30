from dataclasses import dataclass
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.model.model_family import ModelFamily
from mt.infra.observability.runtime.stage_events import (
	log_model_fit_and_predict_horizon_end,
	log_model_fit_and_predict_horizon_start,
)
from mt.infra.backtest.calibration_builder import build_calibration_summary
from mt.infra.backtest.prediction_frame_builder import build_prediction_frame
from mt.infra.probabilistic.conformal import ConformalCalibrator
from mt.domain.probabilistic.probabilistic_settings import (
	DEFAULT_INTERVAL_LEVELS,
	DEFAULT_PROBABILISTIC_QUANTILES,
)

SUPPRESSED_CONVERGENCE_WARNING_MODELS: set[str] = set()


@dataclass(slots=True)
class WindowExecutionResult:
	"""Результат прогона модели по набору окон"""

	frames: list[pd.DataFrame]
	warnings: list[str]
	train_time_seconds: float
	inference_time_seconds: float
	suppressed_convergence_warnings: int = 0
	calibration_summary: pd.DataFrame | None = None
	probabilistic_metadata: dict[str, object] | None = None


@dataclass(slots=True)
class WindowFrames:
	"""Подготовленные фреймы для одного окна и горизонта"""

	train_frame: pd.DataFrame
	valid_predict: pd.DataFrame
	target_column: str


def execute_backtest_windows(
	adapter: ForecastModelAdapter,
	model_name: str,
	prepared_frame: pd.DataFrame,
	feature_columns: list[str],
	windows: pd.DataFrame,
	seed: int,
) -> WindowExecutionResult:
	frames: list[pd.DataFrame] = []
	warnings_list: list[str] = []
	train_time_total = 0.0
	inference_time_total = 0.0
	suppressed_convergence_warnings = 0
	model_info = adapter.get_model_info()
	calibrator = ConformalCalibrator()
	quantile_crossing_corrected = False
	lower_bounds_clipped = False

	for horizon, horizon_windows in iter_horizon_windows(windows):
		log_model_fit_and_predict_horizon_start(model_name, horizon, horizon_windows)

		for window in horizon_windows.itertuples(index=False):
			window_frames = prepare_window_frames(
				adapter=adapter,
				prepared_frame=prepared_frame,
				feature_columns=feature_columns,
				horizon=horizon,
				forecast_origin=window.forecast_origin,
				train_end=window.train_end,
			)
			if window_frames is None:
				raise ValueError()

			predictions, train_elapsed, inference_elapsed, suppressed_warnings = fit_and_predict_window(
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

			prediction_frame, crossing_corrected, clipped = build_prediction_frame(
				adapter=adapter,
				model_info=model_info,
				valid_predict=window_frames.valid_predict,
				target_column=window_frames.target_column,
				forecast_origin=pd.Timestamp(window.forecast_origin),
				target_date=pd.Timestamp(window.test_start),
				horizon=horizon,
				predictions=predictions,
				feature_columns=feature_columns,
				calibrator=calibrator,
			)
			quantile_crossing_corrected = quantile_crossing_corrected or crossing_corrected
			lower_bounds_clipped = lower_bounds_clipped or clipped
			frames.append(prediction_frame)

			calibrator.record_summary(
				horizon=horizon,
				forecast_origin=pd.Timestamp(window.forecast_origin),
				summary=build_calibration_summary(prediction_frame, calibrator, horizon),
			)
			calibrator.update(
				horizon=horizon,
				actual=prediction_frame["actual"].astype(float).to_numpy(),
				prediction=prediction_frame["prediction"].astype(float).to_numpy(),
			)

			log_model_fit_and_predict_horizon_end(
				model_name,
				horizon,
				window.forecast_origin,
				len(prediction_frame),
			)

	return WindowExecutionResult(
		frames=frames,
		warnings=warnings_list,
		train_time_seconds=train_time_total,
		inference_time_seconds=inference_time_total,
		suppressed_convergence_warnings=suppressed_convergence_warnings,
		calibration_summary=calibrator.build_summary_frame(),
		probabilistic_metadata={
			"default_quantiles": list(DEFAULT_PROBABILISTIC_QUANTILES),
			"interval_levels": list(DEFAULT_INTERVAL_LEVELS),
			"native_probabilistic_supported": adapter.supports_native_probabilistic(),
			"quantile_crossing_corrected": quantile_crossing_corrected,
			"lower_bounds_clipped_to_zero": lower_bounds_clipped,
			"conformal_scope": calibrator.config.scope,
			"conformal_min_history": calibrator.config.min_history,
		},
	)


def iter_horizon_windows(windows: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
	return [
		(int(horizon), windows[windows["horizon"] == horizon])
		for horizon in sorted(windows["horizon"].unique())
	]


def prepare_window_frames(
	adapter: ForecastModelAdapter,
	prepared_frame: pd.DataFrame,
	feature_columns: list[str],
	horizon: int,
	forecast_origin: pd.Timestamp,
	train_end: pd.Timestamp,
) -> WindowFrames | None:
	target_column = f"target_h{horizon}"
	if adapter.get_model_info().model_family == ModelFamily.ML:
		target_dates = pd.to_datetime(prepared_frame["week_start"]) + pd.Timedelta(weeks=horizon)
		train_frame = prepared_frame.loc[target_dates <= pd.Timestamp(train_end)].copy()
	else:
		train_frame = prepared_frame.loc[
			pd.to_datetime(prepared_frame["week_start"]) <= pd.Timestamp(train_end)
		].copy()
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


def fit_and_predict_window(
	adapter: ForecastModelAdapter,
	model_name: str,
	window_frames: WindowFrames,
	feature_columns: list[str],
	horizon: int,
	seed: int,
) -> tuple[np.ndarray, float, float, int]:
	train_start = time.perf_counter()
	suppressed_warnings = fit_with_warning_suppression(
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


def fit_with_warning_suppression(
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
