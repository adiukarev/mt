import numpy as np
import pandas as pd

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.probabilistic.probabilistic_settings import DEFAULT_PROBABILISTIC_QUANTILES
from mt.domain.probabilistic.probabilistic import ProbabilisticColumn, ProbabilisticSource, ProbabilisticStatus
from mt.infra.probabilistic.conformal import ConformalCalibrator
from mt.infra.probabilistic.schema import has_complete_probabilistic_output


def resolve_probabilistic_frame(
	adapter: ForecastModelAdapter,
	predict_frame: pd.DataFrame,
	feature_columns: list[str],
	target_column: str,
	horizon: int,
	point_predictions: np.ndarray,
	calibrator: ConformalCalibrator,
	quantiles: tuple[float, ...] = DEFAULT_PROBABILISTIC_QUANTILES,
) -> pd.DataFrame:
	conformal_frame = _resolve_conformal_frame(
		predict_frame=predict_frame,
		horizon=horizon,
		point_predictions=point_predictions,
		calibrator=calibrator,
	)
	if _has_available_probabilistic_frame(conformal_frame):
		return conformal_frame

	if adapter.supports_native_probabilistic():
		native_frame = _resolve_native_probabilistic_frame(
			adapter=adapter,
			predict_frame=predict_frame,
			feature_columns=feature_columns,
			target_column=target_column,
			horizon=horizon,
			quantiles=quantiles,
		)
		if native_frame is not None:
			native_frame["prediction"] = native_frame[ProbabilisticColumn.Q50].astype(float)
			native_frame[ProbabilisticColumn.SOURCE] = ProbabilisticSource.NATIVE
			native_frame[ProbabilisticColumn.STATUS] = ProbabilisticStatus.AVAILABLE
			return native_frame

	return conformal_frame


def _resolve_conformal_frame(
	predict_frame: pd.DataFrame,
	horizon: int,
	point_predictions: np.ndarray,
	calibrator: ConformalCalibrator,
) -> pd.DataFrame:
	if "series_id" in predict_frame.columns:
		conformal_frame, _ = calibrator.predict(
			point_predictions.astype(float),
			horizon,
			series_ids=predict_frame["series_id"].astype(str),
		)
	else:
		conformal_frame, _ = calibrator.predict(point_predictions.astype(float), horizon)
	return conformal_frame


def _has_available_probabilistic_frame(frame: pd.DataFrame) -> bool:
	if frame.empty or ProbabilisticColumn.STATUS not in frame.columns:
		return False
	return bool(
		(
			frame[ProbabilisticColumn.STATUS].astype(str)
			== ProbabilisticStatus.AVAILABLE
		).all()
	)


def _resolve_native_probabilistic_frame(
	adapter: ForecastModelAdapter,
	predict_frame: pd.DataFrame,
	feature_columns: list[str],
	target_column: str,
	horizon: int,
	quantiles: tuple[float, ...],
) -> pd.DataFrame | None:
	try:
		native_frame = adapter.predict_quantiles(
			predict_frame=predict_frame,
			feature_columns=feature_columns,
			target_column=target_column,
			horizon=horizon,
			quantiles=quantiles,
		)
	except Exception:
		return None

	if native_frame is None or len(native_frame) != len(predict_frame):
		return None
	if not has_complete_probabilistic_output(native_frame):
		return None

	return native_frame.copy()
