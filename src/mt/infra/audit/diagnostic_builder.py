import numpy as np
import pandas as pd

from mt.domain.audit.audit_series import (
	compute_coefficient_of_variation,
	compute_outlier_share as calculate_outlier_share,
	compute_sales_scale,
	compute_skewness,
	compute_trend_strength,
	compute_volatility,
	infer_focus_lags,
	safe_autocorr,
	safe_ratio,
	split_mean_levels,
	split_std_levels,
)
from mt.domain.audit.audit_stationarity import calculate_adf, calculate_kpss, infer_stationarity


def build_summary(weekly: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
	aggregations: dict[str, tuple[str, object]] = {
		"start_date": ("week_start", "min"),
		"end_date": ("week_start", "max"),
		"history_weeks": ("week_start", "count"),
		"total_sales_units": ("sales_units", "sum"),
		"mean_sales_units": ("sales_units", "mean"),
		"median_sales_units": ("sales_units", "median"),
		"std_sales_units": ("sales_units", lambda s: float(s.std(ddof=0))),
		"zero_share": ("sales_units", lambda s: float((s == 0).mean())),
		"missing_share": ("sales_units", lambda s: float(s.isna().mean())),
	}
	if "category" in weekly.columns:
		aggregations["category"] = ("category", "first")

	summary = (
		weekly.groupby("series_id", as_index=False)
		.agg(**aggregations)
		.merge(
			segments.drop(columns=[column for column in ("history_weeks", "zero_share") if column in segments.columns]),
			on="series_id",
			how="left",
		)
	)
	summary["expected_weeks"] = (((summary["end_date"] - summary["start_date"]).dt.days // 7) + 1).astype(int)
	summary["grid_coverage"] = (
		summary["history_weeks"] / summary["expected_weeks"].replace(0, np.nan)
	).fillna(0.0)
	summary["weekly_grid_complete"] = summary["grid_coverage"].ge(0.999999)

	return summary.merge(build_series_diagnostics(weekly), on="series_id", how="left")


def build_series_diagnostics(weekly: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, float | str]] = []
	for series_id, group in weekly.groupby("series_id"):
		ordered = group.sort_values("week_start").copy()
		values = ordered["sales_units"].astype(float).reset_index(drop=True)
		complete_week_values = _select_complete_week_values(ordered)
		acf_lag_1 = safe_autocorr(values, 1)
		acf_lag_7 = safe_autocorr(values, 7)
		acf_lag_28 = safe_autocorr(values, 28)
		acf_lag_52 = safe_autocorr(values, 52)
		first_half_mean, second_half_mean = split_mean_levels(values)
		first_half_std, second_half_std = split_std_levels(values)
		rows.append(
			{
				"series_id": series_id,
				"coefficient_of_variation": compute_coefficient_of_variation(values),
				"trend_strength": compute_trend_strength(values),
				"skewness": compute_skewness(values),
				"sales_scale": compute_sales_scale(values),
				"volatility": compute_volatility(values),
				"outlier_share": compute_outlier_share(complete_week_values),
				"mean_shift_ratio": safe_ratio(second_half_mean, first_half_mean),
				"variance_shift_ratio": safe_ratio(second_half_std, first_half_std),
				"acf_focus_lags": infer_focus_lags(acf_lag_1, acf_lag_7, acf_lag_28, acf_lag_52),
			}
		)
	return pd.DataFrame(rows)


def build_segment_summary(summary: pd.DataFrame) -> pd.DataFrame:
	return (
		summary.groupby("segment_label", as_index=False)
		.agg(
			series_count=("series_id", "nunique"),
			share_of_series=("series_id", lambda s: float(s.nunique() / summary["series_id"].nunique())),
			mean_history_weeks=("history_weeks", "mean"),
			mean_zero_share=("zero_share", "mean"),
			mean_outlier_share=("outlier_share", "mean"),
			mean_total_sales=("total_sales_units", "mean"),
		)
		.sort_values("series_count", ascending=False)
		.reset_index(drop=True)
	)


def compute_outlier_share(series: pd.Series) -> float:
	return calculate_outlier_share(series)


def _select_complete_week_values(series_frame: pd.DataFrame) -> pd.Series:
	if "days_in_week" not in series_frame.columns:
		return series_frame["sales_units"].astype(float).reset_index(drop=True)
	complete_weeks = series_frame.loc[series_frame["days_in_week"].eq(7), "sales_units"].astype(float)
	if complete_weeks.empty:
		return series_frame["sales_units"].astype(float).reset_index(drop=True)
	return complete_weeks.reset_index(drop=True)


def build_seasonality_summary(weekly: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for series_id, group in weekly.groupby("series_id"):
		ordered = group.sort_values("week_start").copy()
		sales = ordered["sales_units"].astype(float).reset_index(drop=True)
		rows.append(
			{
				"series_id": series_id,
				"acf_lag_1": safe_autocorr(sales, 1),
				"acf_lag_7": safe_autocorr(sales, 7),
				"acf_lag_28": safe_autocorr(sales, 28),
				"acf_lag_52": safe_autocorr(sales, 52),
			}
		)
	return pd.DataFrame(rows)


def build_stationarity_summary(weekly: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for series_id, group in weekly.groupby("series_id"):
		ordered = group.sort_values("week_start").copy()
		values = ordered["sales_units"].astype(float).reset_index(drop=True)
		adf_stat, adf_pvalue = calculate_adf(values)
		kpss_stat, kpss_pvalue, kpss_pvalue_note = calculate_kpss(values)
		rows.append(
			{
				"series_id": series_id,
				"adf_stat": adf_stat,
				"adf_pvalue": adf_pvalue,
				"kpss_stat": kpss_stat,
				"kpss_pvalue": kpss_pvalue,
				"kpss_pvalue_note": kpss_pvalue_note,
				"stationarity_hint": infer_stationarity(adf_pvalue, kpss_pvalue, kpss_pvalue_note),
			}
		)
	return pd.DataFrame(rows)
