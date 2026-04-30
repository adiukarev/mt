import pandas as pd

from mt.domain.metric.metric_name import PROBABILISTIC_METRIC_NAMES
from mt.domain.metric.metric_point import calculate_point_metrics
from mt.domain.metric.metric_probabilistic import calculate_probabilistic_metrics


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
	if predictions.empty:
		return pd.DataFrame()
	scored_predictions = predictions.loc[predictions["actual"].notna()].copy()
	if scored_predictions.empty:
		return pd.DataFrame()
	rows: list[dict[str, object]] = []
	for horizon, horizon_frame in scored_predictions.groupby("horizon", sort=True):
		rows.append(
			{
				"horizon": int(horizon),
				**calculate_point_metrics(horizon_frame),
			}
		)
	return pd.DataFrame(rows)


def build_probabilistic_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
	if predictions.empty:
		return pd.DataFrame(
			columns=["horizon", *PROBABILISTIC_METRIC_NAMES]
		)
	scored_predictions = predictions.loc[predictions["actual"].notna()].copy()
	if scored_predictions.empty:
		return pd.DataFrame(columns=["horizon", *PROBABILISTIC_METRIC_NAMES])
	rows: list[dict[str, object]] = []
	for horizon, horizon_frame in scored_predictions.groupby("horizon", sort=True):
		rows.append(
			{
				"horizon": int(horizon),
				**calculate_probabilistic_metrics(horizon_frame),
			}
		)
	return pd.DataFrame(rows)
