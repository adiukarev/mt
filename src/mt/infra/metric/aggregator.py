import pandas as pd

from mt.domain.metric.metric_name import (
	POINT_METRIC_NAMES,
	PROBABILISTIC_METRIC_NAMES,
	PROBABILISTIC_METRIC_SORT_ORDER,
)
from mt.domain.metric.metric_point import calculate_point_metrics
from mt.domain.metric.metric_probabilistic import calculate_probabilistic_metrics

METRIC_NAMES = POINT_METRIC_NAMES


def aggregate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	if predictions.empty:
		empty_columns = ["model_name", *METRIC_NAMES]
		return pd.DataFrame(columns=empty_columns), pd.DataFrame(
			columns=["model_name", "horizon", *METRIC_NAMES]
		)

	overall: list[dict[str, object]] = []
	by_horizon: list[dict[str, object]] = []
	for model_name, model_frame in predictions.groupby("model_name"):
		metrics = calculate_point_metrics(model_frame)
		overall.append({"model_name": model_name, **metrics})
		for horizon, horizon_frame in model_frame.groupby("horizon"):
			by_horizon.append(
				{"model_name": model_name, "horizon": horizon, **calculate_point_metrics(horizon_frame)})

	overall_df = pd.DataFrame(overall).sort_values("WAPE").reset_index(drop=True)
	by_horizon_df = pd.DataFrame(by_horizon).sort_values(["model_name", "horizon"]).reset_index(
		drop=True)
	return overall_df, by_horizon_df


def aggregate_probabilistic_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	if predictions.empty:
		empty_columns = ["model_name", *PROBABILISTIC_METRIC_NAMES]
		return pd.DataFrame(columns=empty_columns), pd.DataFrame(
			columns=["model_name", "horizon", *PROBABILISTIC_METRIC_NAMES]
		)

	overall: list[dict[str, object]] = []
	by_horizon: list[dict[str, object]] = []
	for model_name, model_frame in predictions.groupby("model_name"):
		metrics = calculate_probabilistic_metrics(model_frame)
		if all(pd.isna(value) for value in metrics.values()):
			continue
		overall.append({"model_name": model_name, **metrics})
		for horizon, horizon_frame in model_frame.groupby("horizon"):
			horizon_metrics = calculate_probabilistic_metrics(horizon_frame)
			if all(pd.isna(value) for value in horizon_metrics.values()):
				continue
			by_horizon.append({"model_name": model_name, "horizon": horizon, **horizon_metrics})

	overall_df = pd.DataFrame(overall)
	by_horizon_df = pd.DataFrame(by_horizon)
	if overall_df.empty:
		overall_df = pd.DataFrame(columns=["model_name", *PROBABILISTIC_METRIC_NAMES])
	else:
		overall_df = overall_df.sort_values(list(PROBABILISTIC_METRIC_SORT_ORDER),
		                                    na_position="last").reset_index(drop=True)
	if by_horizon_df.empty:
		by_horizon_df = pd.DataFrame(columns=["model_name", "horizon", *PROBABILISTIC_METRIC_NAMES])
	else:
		by_horizon_df = by_horizon_df.sort_values(["model_name", "horizon"]).reset_index(drop=True)
	return overall_df, by_horizon_df
