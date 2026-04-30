import pandas as pd

from mt.infra.audit.feature_builder import default_feature_manifest
from mt.infra.feature.supervised_builder import build_supervised_frame

_SNAPSHOT_COLUMNS = [
	"series_id",
	"category",
	"week_start",
	"sales_units",
	"segment_label",
	"lag_1",
	"lag_7",
	"lag_28",
	"rolling_mean_7",
	"rolling_median_7",
	"rolling_mad_7",
	"rolling_iqr_7",
	"rolling_robust_zscore_7",
	"rolling_recent_outlier_flag_7",
	"rolling_std_7",
	"rolling_max_7",
	"rolling_min_7",
	"rolling_mean_28",
	"rolling_median_28",
	"rolling_mad_28",
	"rolling_iqr_28",
	"rolling_robust_zscore_28",
	"rolling_recent_outlier_flag_28",
	"rolling_ratio_to_mean_7",
	"rolling_recent_trend_delta_28",
	"week_of_year",
	"month",
	"quarter",
	"category_code",
	"segment_code",
]


def build_series_feature_snapshots(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
	if weekly.empty:
		return {}
	selected_series_ids = (
		weekly.loc[:, "series_id"]
		.astype(str)
		.drop_duplicates()
		.sort_values()
		.tolist()
	)
	selected_weekly = weekly.loc[weekly["series_id"].astype(str).isin(selected_series_ids)].copy()
	selected_segments = segments.loc[segments["series_id"].astype(str).isin(selected_series_ids)].copy()
	supervised, _ = build_supervised_frame(
		selected_weekly,
		selected_segments,
		default_feature_manifest(),
	)
	available_columns = [column for column in _SNAPSHOT_COLUMNS if column in supervised.columns]
	snapshots: dict[str, pd.DataFrame] = {}
	for series_id in selected_series_ids:
		snapshot = supervised.loc[
			supervised["series_id"].astype(str) == series_id, available_columns
		].copy()
		required_anchor = "lag_28" if "lag_28" in snapshot.columns else "lag_1"
		snapshot = snapshot.dropna(subset=[required_anchor], how="any")
		if snapshot.empty:
			snapshot = supervised.loc[
				supervised["series_id"].astype(str) == series_id, available_columns
			].copy()
		snapshots[series_id] = snapshot.tail(20).reset_index(drop=True)

	return snapshots
