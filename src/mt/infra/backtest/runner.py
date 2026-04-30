import time

import pandas as pd

from mt.domain.model.model_result import ModelResult
from mt.domain.model.model_config_manifest import ModelConfigManifest
from mt.infra.backtest.windows_executor import execute_backtest_windows
from mt.infra.model.adapter_builder import build_model_adapter
from mt.infra.probabilistic.schema import (
	build_empty_prediction_frame,
	finalize_prediction_frame,
)


def run_backtest(
	model_name: str,
	supervised: pd.DataFrame,
	feature_columns: list[str],
	windows: pd.DataFrame,
	seed: int,
	config: ModelConfigManifest | dict[str, object] | None = None,
) -> ModelResult:
	"""Прогнать одну модель по всем rolling-окнам backtesting"""

	model_wall_start = time.perf_counter()

	adapter = build_model_adapter(model_name, config)
	prepared_frame = adapter.prepare_frame(supervised)
	resolved_feature_columns = adapter.resolve_feature_columns(prepared_frame, feature_columns)

	result = execute_backtest_windows(
		adapter=adapter,
		model_name=model_name,
		prepared_frame=prepared_frame,
		feature_columns=resolved_feature_columns,
		windows=windows,
		seed=seed,
	)

	predictions = (
		pd.concat(result.frames, ignore_index=True)
		if result.frames
		else build_empty_prediction_frame()
	)
	if not predictions.empty and predictions[["actual", "prediction"]].isna().any().any():
		raise ValueError()

	predictions, crossing_corrected, clipped = finalize_prediction_frame(predictions)
	probabilistic_metadata = dict(result.probabilistic_metadata or {})
	probabilistic_metadata["quantile_crossing_corrected"] = (
		bool(probabilistic_metadata.get("quantile_crossing_corrected", False)) or crossing_corrected
	)
	probabilistic_metadata["lower_bounds_clipped_to_zero"] = (
		bool(probabilistic_metadata.get("lower_bounds_clipped_to_zero", False)) or clipped
	)

	return ModelResult(
		info=adapter.get_model_info(),
		predictions=predictions,
		warnings=sorted(set(result.warnings)),
		train_time_seconds=result.train_time_seconds,
		inference_time_seconds=result.inference_time_seconds,
		wall_time_seconds=time.perf_counter() - model_wall_start,
		used_feature_columns=resolved_feature_columns,
		calibration_summary=result.calibration_summary,
		probabilistic_metadata=probabilistic_metadata,
	)
