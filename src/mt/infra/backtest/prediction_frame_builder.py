import numpy as np
import pandas as pd

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.model.model_info import ModelInfo
from mt.infra.probabilistic.conformal import ConformalCalibrator
from mt.infra.probabilistic.inference import resolve_probabilistic_frame
from mt.infra.probabilistic.schema import finalize_prediction_frame


def build_prediction_frame(
	adapter: ForecastModelAdapter,
	model_info: ModelInfo,
	valid_predict: pd.DataFrame,
	target_column: str,
	forecast_origin: pd.Timestamp,
	target_date: pd.Timestamp,
	horizon: int,
	predictions: np.ndarray,
	feature_columns: list[str],
	calibrator: ConformalCalibrator,
) -> tuple[pd.DataFrame, bool, bool]:
	base = pd.DataFrame(
		{
			"model_name": model_info.model_name,
			"model_family": model_info.model_family,
			"series_id": valid_predict["series_id"].astype(str).to_numpy(),
			"category": valid_predict["category"].to_numpy(),
			"segment_label": valid_predict.get("segment_label"),
			"forecast_origin": forecast_origin,
			"target_date": target_date,
			"horizon": horizon,
			"actual": valid_predict[target_column].astype(float).to_numpy(),
			"prediction": predictions.astype(float),
		}
	)

	probabilistic_frame = resolve_probabilistic_frame(
		adapter=adapter,
		predict_frame=valid_predict,
		feature_columns=feature_columns,
		target_column=target_column,
		horizon=horizon,
		point_predictions=predictions.astype(float),
		calibrator=calibrator,
	)
	for column in probabilistic_frame.columns:
		base[column] = probabilistic_frame[column].to_numpy()

	return finalize_prediction_frame(base)
