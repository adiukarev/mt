import pandas as pd

from mt.infra.metric.calculates import calculate_metrics

METRIC_NAMES: tuple[str, ...] = ("WAPE", "sMAPE", "MAE", "RMSE", "Bias", "MedianAE")


def extract_actual_and_prediction(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
	"""Достать согласованные серии actual/prediction из prediction frame."""

	return frame["actual"].astype(float), frame["prediction"].astype(float)


def aggregate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Агрегировать метрики по моделям и горизонтам"""

	if predictions.empty:
		empty_columns = ["model_name", *METRIC_NAMES]
		return pd.DataFrame(columns=empty_columns), pd.DataFrame(
			columns=["model_name", "horizon", *METRIC_NAMES])

	overall = []
	by_horizon = []
	for model_name, model_frame in predictions.groupby("model_name"):
		metrics = calculate_metrics(model_frame)
		overall.append({"model_name": model_name, **metrics})
		for horizon, horizon_frame in model_frame.groupby("horizon"):
			by_horizon.append(
				{
					"model_name": model_name,
					"horizon": horizon,
					**calculate_metrics(horizon_frame),
				}
			)

	overall_df = pd.DataFrame(overall).sort_values("WAPE").reset_index(drop=True)
	by_horizon_df = pd.DataFrame(by_horizon).sort_values(["model_name", "horizon"]).reset_index(
		drop=True)

	return overall_df, by_horizon_df
