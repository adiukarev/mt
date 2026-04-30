import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle


def build_forecast_frame(
	dataset: DatasetBundle,
	horizon_weeks: int,
) -> pd.DataFrame:
	if horizon_weeks < 1:
		raise ValueError("horizon_weeks must be positive")

	frame = dataset.weekly.copy()
	if frame.empty:
		raise ValueError("Dataset bundle is empty")

	required_columns = {"series_id", "category", "week_start", "sales_units"}
	missing = required_columns.difference(frame.columns)
	if missing:
		raise ValueError(f"Weekly bundle is missing columns: {sorted(missing)}")

	frame["week_start"] = pd.to_datetime(frame["week_start"], utc=False)
	frame = _drop_incomplete_edge_weeks(frame)

	max_week_start = frame["week_start"].max()
	forecast_origin = pd.Timestamp(max_week_start) - pd.Timedelta(weeks=horizon_weeks)
	frame["is_history"] = frame["week_start"] <= forecast_origin

	if bool(frame["is_history"].all()) or bool((~frame["is_history"]).all()):
		raise ValueError("Unable to split weekly bundle into history and future forecast window")

	return frame.sort_values(["series_id", "week_start"]).reset_index(drop=True)


def infer_horizon(frame: pd.DataFrame) -> int:
	horizon = int((~frame["is_history"].astype(bool)).sum() / frame["series_id"].nunique())
	if horizon < 1:
		raise ValueError("Unable to infer positive forecast horizon")
	return horizon


def _drop_incomplete_edge_weeks(frame: pd.DataFrame) -> pd.DataFrame:
	if "days_in_week" not in frame.columns:
		return frame

	week_days = (
		frame.groupby("week_start", as_index=False)["days_in_week"]
		.min()
		.sort_values("week_start")
		.reset_index(drop=True)
	)
	if week_days.empty:
		return frame

	full_week_days = int(week_days["days_in_week"].max())
	if full_week_days <= 0:
		return frame

	complete_mask = week_days["days_in_week"].astype(int) >= full_week_days
	if complete_mask.all():
		return frame
	if not complete_mask.any():
		raise ValueError("Weekly bundle contains no complete weeks")

	first_complete = week_days.loc[complete_mask, "week_start"].iloc[0]
	last_complete = week_days.loc[complete_mask, "week_start"].iloc[-1]
	return frame.loc[
		(frame["week_start"] >= first_complete)
		& (frame["week_start"] <= last_complete)
	].copy()
