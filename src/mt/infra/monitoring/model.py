from dataclasses import asdict
from typing import Any

import pandas as pd

from mt.domain.monitoring.monitoring_reference_model import (
	MonitoringReferenceModelState,
	MonitoringReferenceModelType,
)


def train_monitoring_reference_model(
	frame: pd.DataFrame,
	model_type: MonitoringReferenceModelType,
	lookback_weeks: int,
) -> MonitoringReferenceModelState:
	history = frame.sort_values(["series_id", "week_start"]).reset_index(drop=True)
	grouped = history.groupby("series_id", sort=False)
	series_baseline: dict[str, float] = {}
	for series_id, series_frame in grouped:
		tail = series_frame.tail(lookback_weeks)
		if model_type == MonitoringReferenceModelType.ROLLING_MEAN:
			value = float(tail["sales_units"].mean())
		else:
			value = float(tail["sales_units"].iloc[-1])
		series_baseline[str(series_id)] = value

	global_tail = history.tail(max(len(series_baseline), 1))
	global_baseline = float(global_tail["sales_units"].mean()) if not global_tail.empty else 0.0

	return MonitoringReferenceModelState(
		model_type=model_type,
		lookback_weeks=lookback_weeks,
		trained_on_rows=int(len(history)),
		trained_until=str(pd.Timestamp(history["week_start"].max()).date()) if not history.empty else None,
		global_baseline=global_baseline,
		series_baseline=series_baseline,
	)


def predict_recent_actuals(
	reference_frame: pd.DataFrame,
	recent_actuals: pd.DataFrame,
	model_type: MonitoringReferenceModelType,
	lookback_weeks: int,
) -> pd.DataFrame:
	if recent_actuals.empty:
		return pd.DataFrame(
			columns=[
				"series_id",
				"category",
				"forecast_origin",
				"target_date",
				"horizon",
				"actual",
				"prediction",
				"model_name",
			]
		)

	history = reference_frame.sort_values(["series_id", "week_start"]).reset_index(drop=True)
	rows: list[dict[str, Any]] = []
	for week_start, actual_frame in recent_actuals.groupby("week_start", sort=True):
		state = train_monitoring_reference_model(
			history,
			model_type=model_type,
			lookback_weeks=lookback_weeks,
		)
		forecast_origin = pd.Timestamp(week_start) - pd.Timedelta(weeks=1)
		for row in actual_frame.itertuples(index=False):
			prediction = state.series_baseline.get(str(row.series_id), state.global_baseline)
			rows.append(
				{
					"series_id": str(row.series_id),
					"category": getattr(row, "category", None),
					"scenario_name": getattr(row, "scenario_name", None),
					"forecast_origin": forecast_origin,
					"target_date": pd.Timestamp(week_start),
					"horizon": 1,
					"actual": float(row.sales_units),
					"prediction": float(prediction),
					"model_name": state.model_type.value,
				}
			)
		history = pd.concat([history, actual_frame], ignore_index=True)

	return pd.DataFrame(rows).sort_values(["target_date", "series_id"]).reset_index(drop=True)


def serialize_model_state(state: MonitoringReferenceModelState) -> dict[str, Any]:
	return asdict(state)
