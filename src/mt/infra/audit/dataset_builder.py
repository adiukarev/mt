import numpy as np
import pandas as pd

from mt.domain.series_segmentation.series_segmentation_label import SegmentLabel


def build_dataset_profile(
	weekly: pd.DataFrame,
	summary: pd.DataFrame,
	segments: pd.DataFrame,
	metadata: dict[str, object],
	aggregation_level: str,
	raw_context: dict[str, object],
) -> pd.DataFrame:
	rows = [
		{"metric": "aggregation_level", "value": aggregation_level},
		{"metric": "source_frequency", "value": metadata.get("source_frequency", "unknown")},
		{"metric": "weekly_rule", "value": metadata.get("weekly_rule", "unknown")},
		{"metric": "period_start", "value": str(weekly["week_start"].min().date())},
		{"metric": "period_end", "value": str(weekly["week_start"].max().date())},
		{"metric": "number_of_series", "value": int(weekly["series_id"].nunique())},
		{
			"metric": "number_of_categories",
			"value": int(weekly["category"].nunique()) if "category" in weekly.columns else 0,
		},
		{"metric": "history_weeks_min", "value": int(summary["history_weeks"].min())},
		{"metric": "history_weeks_median", "value": float(summary["history_weeks"].median())},
		{"metric": "history_weeks_max", "value": int(summary["history_weeks"].max())},
		{"metric": "zero_share_mean", "value": float(summary["zero_share"].mean())},
		{"metric": "missing_share_mean", "value": float(summary["missing_share"].mean())},
		{"metric": "outlier_share_mean", "value": float(summary["outlier_share"].mean())},
		{"metric": "mean_cv", "value": float(summary["coefficient_of_variation"].mean())},
		{"metric": "mean_trend_strength", "value": float(summary["trend_strength"].mean())},
		{"metric": "short_history_share", "value": float(summary["short_history"].mean())},
		{"metric": "high_zero_share_share", "value": float(summary["high_zero_share"].mean())},
		{"metric": "high_variance_share", "value": float(summary["high_variance"].mean())},
		{
			"metric": "problematic_share",
			"value": float(segments["segment_label"].eq(SegmentLabel.PROBLEMATIC).mean()),
		},
		{
			"metric": "intermittent_share",
			"value": float(segments["segment_label"].eq(SegmentLabel.INTERMITTENT).mean()),
		},
		{"metric": "weekly_grid_complete_share",
		 "value": float(summary["weekly_grid_complete"].mean())},
		{"metric": "stockout_risk", "value": metadata.get("stockout_risk", "unknown")},
		{
			"metric": "structural_shift_risk",
			"value": metadata.get("structural_shift_risk", "not_assessed"),
		},
	]
	for key in (
			"raw_item_count",
			"raw_store_count",
			"raw_state_count",
			"raw_department_count",
			"raw_daily_observations",
	):
		if key in raw_context:
			rows.append({"metric": key, "value": raw_context[key]})
	return pd.DataFrame(rows)


def build_category_summary(
	weekly: pd.DataFrame,
	summary: pd.DataFrame,
	raw_context: dict[str, object],
) -> pd.DataFrame:
	if "category" not in weekly.columns or "category" not in summary.columns:
		return pd.DataFrame()
	category_summary = (
		weekly.groupby("category", as_index=False)
		.agg(
			number_of_series=("series_id", "nunique"),
			total_sales_units=("sales_units", "sum"),
			mean_weekly_sales=("sales_units", "mean"),
			median_weekly_sales=("sales_units", "median"),
		)
		.merge(
			summary.groupby("category", as_index=False).agg(
				mean_zero_share=("zero_share", "mean"),
				mean_outlier_share=("outlier_share", "mean"),
			),
			on="category",
			how="left",
		)
	)
	item_counts = raw_context.get("item_counts_by_category")
	if isinstance(item_counts, pd.DataFrame) and not item_counts.empty:
		category_summary = category_summary.merge(item_counts, on="category", how="left")

	return category_summary.sort_values("total_sales_units", ascending=False).reset_index(drop=True)


def build_sku_summary(summary: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
	if aggregation_level != "sku" or summary.empty:
		return pd.DataFrame()

	ordered = summary.sort_values(
		["total_sales_units", "mean_sales_units", "series_id"],
		ascending=[False, False, True],
	).reset_index(drop=True)
	total_sales = float(ordered["total_sales_units"].sum())
	ordered["sales_rank"] = np.arange(1, len(ordered) + 1)
	ordered["sales_share"] = ordered["total_sales_units"] / total_sales if total_sales > 0 else 0.0
	ordered["cumulative_sales_share"] = ordered["sales_share"].cumsum()

	return ordered[
		[
			"sales_rank",
			"series_id",
			"category",
			"segment_label",
			"total_sales_units",
			"sales_share",
			"cumulative_sales_share",
			"mean_sales_units",
			"median_sales_units",
			"zero_share",
			"outlier_share",
			"coefficient_of_variation",
			"trend_strength",
		]
	]
