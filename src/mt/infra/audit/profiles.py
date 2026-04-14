import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from mt.domain.manifest import FeatureManifest
from mt.infra.feature.registry import build_feature_registry
from mt.infra.feature.supervised_builder import make_supervised_frame


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
		"outlier_share": ("sales_units", lambda s: float(compute_outlier_share(s.astype(float)))),
	}
	if "category" in weekly.columns:
		aggregations["category"] = ("category", "first")

	summary = (
		weekly.groupby("series_id", as_index=False)
		.agg(**aggregations)
		.merge(
			segments.drop(
				columns=[column for column in ("history_weeks", "zero_share") if column in segments.columns]
			),
			on="series_id",
			how="left",
		)
	)
	summary["expected_weeks"] = (
		((summary["end_date"] - summary["start_date"]).dt.days // 7) + 1
	).astype(int)
	summary["grid_coverage"] = (
		summary["history_weeks"] / summary["expected_weeks"].replace(0, np.nan)
	).fillna(0.0)
	summary["weekly_grid_complete"] = summary["grid_coverage"].ge(0.999999)

	return summary.merge(build_series_diagnostics(weekly), on="series_id", how="left")


def compute_outlier_share(series: pd.Series) -> float:
	values = series.dropna().astype(float)
	if len(values) <= 1:
		return 0.0
	std_value = float(values.std(ddof=0))
	if std_value == 0.0:
		return 0.0
	return float(((values - values.median()).abs() > 3 * std_value).mean())


def build_series_diagnostics(weekly: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, float | str]] = []
	for series_id, group in weekly.groupby("series_id"):
		ordered = group.sort_values("week_start").copy()
		values = ordered["sales_units"].astype(float).reset_index(drop=True)
		mean_value = float(values.mean()) if len(values) else 0.0
		std_value = float(values.std(ddof=0)) if len(values) else 0.0
		rows.append(
			{
				"series_id": series_id,
				"coefficient_of_variation": std_value / mean_value if mean_value else float("nan"),
				"trend_strength": compute_trend_strength(values),
			}
		)
	return pd.DataFrame(rows)


def compute_trend_strength(values: pd.Series) -> float:
	if len(values) <= 1:
		return 0.0
	x = np.arange(len(values), dtype=float)
	y = values.to_numpy(dtype=float)
	y_std = float(np.std(y, ddof=0))
	if y_std == 0.0:
		return 0.0
	slope, _ = np.polyfit(x, y, deg=1)
	return float(abs(slope) * len(values) / y_std)


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
		{"metric": "number_of_categories",
		 "value": int(weekly["category"].nunique()) if "category" in weekly.columns else 0},
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
		{"metric": "problematic_share",
		 "value": float(segments["segment_label"].eq("problematic").mean())},
		{"metric": "intermittent_share",
		 "value": float(segments["segment_label"].eq("intermittent").mean())},
		{"metric": "weekly_grid_complete_share",
		 "value": float(summary["weekly_grid_complete"].mean())},
		{"metric": "price_available_raw",
		 "value": bool(raw_context.get("price_available_raw", metadata.get("price_available", False)))},
		{"metric": "promo_available_raw",
		 "value": bool(raw_context.get("promo_available_raw", metadata.get("promo_available", False)))},
		{"metric": "price_allowed_for_default_model",
		 "value": bool(raw_context.get("price_allowed_for_default_model", False))},
		{"metric": "promo_allowed_for_default_model",
		 "value": bool(raw_context.get("promo_allowed_for_default_model", False))},
		{"metric": "stockout_risk", "value": metadata.get("stockout_risk", "unknown")},
		{"metric": "structural_shift_risk",
		 "value": metadata.get("structural_shift_risk", "not_assessed")},
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


def build_aggregation_comparison(
	weekly: pd.DataFrame,
	raw_context: dict[str, object],
) -> pd.DataFrame:
	series_frames: list[tuple[str, pd.DataFrame]] = []
	daily_total = raw_context.get("daily_total_sales")
	if isinstance(daily_total, pd.DataFrame) and not daily_total.empty:
		series_frames.append(
			(
				"daily",
				daily_total.loc[:, ["date", "sales_units"]].rename(columns={"date": "timestamp"}),
			)
		)

	weekly_total = (
		weekly.groupby("week_start", as_index=False)["sales_units"]
		.sum()
		.rename(columns={"week_start": "timestamp"})
	)
	series_frames.append(("weekly", weekly_total))

	monthly_total = (
		weekly.assign(month_start=weekly["week_start"].dt.to_period("M").dt.to_timestamp())
		.groupby("month_start", as_index=False)["sales_units"]
		.sum()
		.rename(columns={"month_start": "timestamp"})
	)
	series_frames.append(("monthly", monthly_total))

	rows: list[dict[str, object]] = []
	for frequency, frame in series_frames:
		values = frame["sales_units"].astype(float)
		mean_value = float(values.mean()) if len(values) else float("nan")
		std_value = float(values.std(ddof=0)) if len(values) else float("nan")
		rows.append(
			{
				"frequency": frequency,
				"period_count": int(len(frame)),
				"period_start": str(frame["timestamp"].min().date()) if len(frame) else "",
				"period_end": str(frame["timestamp"].max().date()) if len(frame) else "",
				"mean_sales_units": mean_value,
				"std_sales_units": std_value,
				"coefficient_of_variation": std_value / mean_value if mean_value else float("nan"),
				"first_order_autocorr": safe_autocorr(values.reset_index(drop=True), 1),
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


def build_sku_summary(
	summary: pd.DataFrame,
	aggregation_level: str,
) -> pd.DataFrame:
	if aggregation_level != "sku" or summary.empty:
		return pd.DataFrame()
	ordered = summary.sort_values(
		["total_sales_units", "mean_sales_units", "series_id"],
		ascending=[False, False, True],
	).reset_index(drop=True)
	total_sales = float(ordered["total_sales_units"].sum())
	ordered["sales_rank"] = np.arange(1, len(ordered) + 1)
	ordered["sales_share"] = (
		ordered["total_sales_units"] / total_sales if total_sales > 0 else 0.0
	)
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


def build_category_correlation_matrix(weekly: pd.DataFrame) -> pd.DataFrame:
	if "category" not in weekly.columns or weekly.empty:
		return pd.DataFrame()

	category_weekly = (
		weekly.groupby(["week_start", "category"], as_index=False)["sales_units"]
		.sum()
		.pivot(index="week_start", columns="category", values="sales_units")
		.sort_index(axis=1)
	)
	if category_weekly.empty or category_weekly.shape[1] < 2:
		return pd.DataFrame()

	return category_weekly.corr().reset_index().rename(columns={"category": "category_name"})


def build_category_growth_correlation_matrix(weekly: pd.DataFrame) -> pd.DataFrame:
	if "category" not in weekly.columns or weekly.empty:
		return pd.DataFrame()

	category_weekly = (
		weekly.groupby(["week_start", "category"], as_index=False)["sales_units"]
		.sum()
		.pivot(index="week_start", columns="category", values="sales_units")
		.sort_index(axis=1)
	)
	if category_weekly.empty or category_weekly.shape[1] < 2:
		return pd.DataFrame()

	log_growth = np.log1p(category_weekly).diff().dropna(how="all")
	if log_growth.empty:
		return pd.DataFrame()
	return log_growth.corr().reset_index().rename(columns={"category": "category_name"})


def build_category_seasonal_index(weekly: pd.DataFrame) -> pd.DataFrame:
	if "category" not in weekly.columns or weekly.empty:
		return pd.DataFrame()

	seasonal = weekly.copy()
	seasonal["week_of_year"] = seasonal["week_start"].dt.isocalendar().week.astype(int)
	seasonal = (
		seasonal.groupby(["category", "week_of_year"], as_index=False)["sales_units"]
		.mean()
		.rename(columns={"sales_units": "mean_sales_units"})
	)
	category_means = (
		weekly.groupby("category", as_index=False)["sales_units"]
		.mean()
		.rename(columns={"sales_units": "category_mean_sales_units"})
	)
	seasonal = seasonal.merge(category_means, on="category", how="left")
	seasonal["seasonal_index"] = (
		seasonal["mean_sales_units"] / seasonal["category_mean_sales_units"].replace(0, np.nan)
	)
	return seasonal.sort_values(["category", "week_of_year"]).reset_index(drop=True)


def build_sku_concentration_summary(
	weekly: pd.DataFrame,
	aggregation_level: str,
) -> pd.DataFrame:
	if aggregation_level != "sku" or weekly.empty:
		return pd.DataFrame()

	sku_totals = (
		weekly.groupby("series_id", as_index=False)["sales_units"]
		.sum()
	)
	if sku_totals.empty:
		return pd.DataFrame()

	sku_totals = sku_totals.sort_values(["sales_units", "series_id"], ascending=[False, True]).reset_index(drop=True)
	total_sales = float(sku_totals["sales_units"].sum())
	if total_sales <= 0:
		return pd.DataFrame()

	shares = sku_totals["sales_units"] / total_sales
	hhi = float(np.square(shares).sum())
	gini = _compute_gini(shares.to_numpy(dtype=float))
	top_limits = [1, 5, 10, 20, 50, 100]
	rows = [
		{"metric": "sku_count", "value": float(len(sku_totals))},
		{"metric": "effective_sku_count_inverse_hhi", "value": float(1.0 / hhi) if hhi > 0 else float("nan")},
		{"metric": "hhi_sales_concentration", "value": hhi},
		{"metric": "gini_sales_concentration", "value": gini},
	]
	for limit in top_limits:
		if len(sku_totals) >= limit:
			rows.append(
				{
					"metric": f"top_{limit}_sales_share",
					"value": float(shares.head(limit).sum()),
				}
			)
	return pd.DataFrame(rows)


def build_sku_share_stability_summary(
	weekly: pd.DataFrame,
	aggregation_level: str,
	top_n: int = 50,
) -> pd.DataFrame:
	if aggregation_level != "sku" or weekly.empty or "category" not in weekly.columns:
		return pd.DataFrame()

	sku_totals = (
		weekly.groupby(["series_id", "category"], as_index=False)["sales_units"]
		.sum()
		.sort_values(["sales_units", "series_id"], ascending=[False, True])
		.head(top_n)
	)
	if sku_totals.empty:
		return pd.DataFrame()

	selected = weekly.loc[weekly["series_id"].isin(sku_totals["series_id"]), ["week_start", "series_id", "category", "sales_units"]].copy()
	selected["category_total"] = selected.groupby(["week_start", "category"])["sales_units"].transform("sum")
	selected["weekly_category_share"] = selected["sales_units"] / selected["category_total"].replace(0, np.nan)

	return (
		selected.groupby(["series_id", "category"], as_index=False)
		.agg(
			total_sales_units=("sales_units", "sum"),
			mean_weekly_share=("weekly_category_share", "mean"),
			std_weekly_share=("weekly_category_share", lambda s: float(s.std(ddof=0))),
			min_weekly_share=("weekly_category_share", "min"),
			max_weekly_share=("weekly_category_share", "max"),
			zero_share=("sales_units", lambda s: float((s == 0).mean())),
		)
		.assign(
			share_cv=lambda df: df["std_weekly_share"] / df["mean_weekly_share"].replace(0, np.nan)
		)
		.sort_values(["share_cv", "total_sales_units"], ascending=[False, False])
		.reset_index(drop=True)
	)


def build_feature_availability(
	metadata: dict[str, object],
	raw_context: dict[str, object],
) -> pd.DataFrame:
	rows = [
		{
			"feature_group": "lags",
			"available_in_raw_data": True,
			"allowed_for_default_modeling": True,
			"reason": "Calculated strictly from past sales history.",
		},
		{
			"feature_group": "rolling",
			"available_in_raw_data": True,
			"allowed_for_default_modeling": True,
			"reason": "Rolling windows remain leakage-safe when shifted to past weeks only.",
		},
		{
			"feature_group": "calendar",
			"available_in_raw_data": True,
			"allowed_for_default_modeling": True,
			"reason": "Calendar is known in advance and does not leak target weeks.",
		},
		{
			"feature_group": "category_encodings",
			"available_in_raw_data": True,
			"allowed_for_default_modeling": True,
			"reason": "Static hierarchy fields are known at forecast origin.",
		},
		{
			"feature_group": "price",
			"available_in_raw_data": bool(
				raw_context.get("price_available_raw", metadata.get("price_available", False))),
			"allowed_for_default_modeling": bool(
				raw_context.get("price_allowed_for_default_model", False)),
			"reason": str(raw_context.get("price_policy_reason",
			                              "Future price values are excluded until forecast-time availability is proven.")),
		},
		{
			"feature_group": "promo",
			"available_in_raw_data": bool(
				raw_context.get("promo_available_raw", metadata.get("promo_available", False))),
			"allowed_for_default_modeling": bool(
				raw_context.get("promo_allowed_for_default_model", False)),
			"reason": str(raw_context.get("promo_policy_reason",
			                              "Promo data is absent or not proven available at forecast origin.")),
		},
		{
			"feature_group": "external",
			"available_in_raw_data": False,
			"allowed_for_default_modeling": False,
			"reason": "No external source is registered with publication timestamps and stable access policy.",
		},
	]
	return pd.DataFrame(rows)


def build_feature_block_summary(aggregation_level: str) -> pd.DataFrame:
	registry = build_feature_registry(
		FeatureManifest(
			enabled=True,
			feature_set="F4",
			lags=[1, 2, 3, 4, 8, 12, 13, 26, 52],
			rolling_windows=[4, 8, 13, 26],
			use_calendar=True,
			use_category_encodings=True,
			use_price=False,
			use_promo=False,
			use_external=False,
		),
		aggregation_level=aggregation_level,
	)
	if registry.empty:
		return pd.DataFrame()

	def _examples(series: pd.Series) -> str:
		return ", ".join(series.head(5).astype(str).tolist())

	grouped = (
		registry.groupby("group", as_index=False)
		.agg(
			feature_count=("name", "count"),
			enabled_count=("enabled", "sum"),
			covariate_classes=("covariate_class", lambda s: ", ".join(sorted(set(s.astype(str))))),
			example_features=("name", _examples),
			availability_at_forecast_time=("availability_at_forecast_time", "all"),
			mechanism_note=("expected_effect_mechanism", "first"),
		)
	)
	grouped["enabled_for_default_modeling"] = grouped["enabled_count"].eq(grouped["feature_count"])
	return grouped[
		[
			"group",
			"feature_count",
			"enabled_count",
			"enabled_for_default_modeling",
			"covariate_classes",
			"availability_at_forecast_time",
			"example_features",
			"mechanism_note",
		]
	]


def build_seasonality_summary(weekly: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for series_id, group in weekly.groupby("series_id"):
		ordered = group.sort_values("week_start").copy()
		sales = ordered["sales_units"].astype(float).reset_index(drop=True)
		rows.append(
			{
				"series_id": series_id,
				"acf_lag_1": safe_autocorr(sales, 1),
				"acf_lag_4": safe_autocorr(sales, 4),
				"acf_lag_8": safe_autocorr(sales, 8),
				"acf_lag_13": safe_autocorr(sales, 13),
				"acf_lag_26": safe_autocorr(sales, 26),
				"acf_lag_52": safe_autocorr(sales, 52),
			}
		)
	return pd.DataFrame(rows)


def build_diagnostic_summary(
	summary: pd.DataFrame,
	seasonality_summary: pd.DataFrame,
) -> pd.DataFrame:
	if summary.empty:
		return pd.DataFrame(columns=["diagnostic", "value", "interpretation"])

	rows = [
		{
			"diagnostic": "mean_history_weeks",
			"value": float(summary["history_weeks"].mean()),
			"interpretation": "Длина истории определяет, можно ли оценивать сезонность и строить rolling backtesting",
		},
		{
			"diagnostic": "mean_zero_share",
			"value": float(summary["zero_share"].mean()),
			"interpretation": "Доля нулей показывает риск прерывистого спроса и ограничение для среднеквадратичных моделей",
		},
		{
			"diagnostic": "mean_cv",
			"value": float(summary["coefficient_of_variation"].mean()),
			"interpretation": "Коэффициент вариации характеризует относительную шумность ряда",
		},
		{
			"diagnostic": "mean_trend_strength",
			"value": float(summary["trend_strength"].mean()),
			"interpretation": "Нормированная сила линейного тренда позволяет понять, достаточны ли чисто сезонные бейзлайн",
		},
		{
			"diagnostic": "seasonal_acf_lag_52_mean",
			"value": float(
				seasonality_summary["acf_lag_52"].mean()) if not seasonality_summary.empty else float(
				"nan"),
			"interpretation": "Автокорреляция на 52 неделях проверяет гипотезу о годовой сезонной памяти",
		},
		{
			"diagnostic": "seasonal_acf_lag_13_mean",
			"value": float(
				seasonality_summary["acf_lag_13"].mean()) if not seasonality_summary.empty else float(
				"nan"),
			"interpretation": "Автокорреляция на 13 неделях проверяет квартальную сезонную память недельных рядов",
		},
		{
			"diagnostic": "seasonal_acf_lag_26_mean",
			"value": float(
				seasonality_summary["acf_lag_26"].mean()) if not seasonality_summary.empty else float(
				"nan"),
			"interpretation": "Автокорреляция на 26 неделях помогает оценить полугодовую память и силу средних сезонных циклов",
		},
		{
			"diagnostic": "strong_seasonality_share_lag_52",
			"value": float(
				seasonality_summary["acf_lag_52"].fillna(0.0).ge(0.30).mean()
			) if not seasonality_summary.empty else float("nan"),
			"interpretation": "Доля рядов с заметной годовой сезонной памятью показывает, насколько Seasonal Naive обязателен как baseline",
		},
	]
	return pd.DataFrame(rows)


def build_stationarity_summary(weekly: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for series_id, group in weekly.groupby("series_id"):
		ordered = group.sort_values("week_start").copy()
		values = ordered["sales_units"].astype(float).reset_index(drop=True)
		adf_stat, adf_pvalue = _safe_adf(values)
		kpss_stat, kpss_pvalue = _safe_kpss(values)
		rows.append(
			{
				"series_id": series_id,
				"adf_stat": adf_stat,
				"adf_pvalue": adf_pvalue,
				"kpss_stat": kpss_stat,
				"kpss_pvalue": kpss_pvalue,
				"stationarity_hint": _infer_stationarity(adf_pvalue, kpss_pvalue),
			}
		)
	return pd.DataFrame(rows)


def build_data_dictionary(weekly: pd.DataFrame) -> pd.DataFrame:
	descriptions = {
		"series_id": "Идентификатор временного ряда на выбранном уровне агрегации",
		"category": "Категория товара в иерархии M5",
		"week_start": "Дата начала недели с якорем W-MON",
		"sales_units": "Целевая переменная: суммарные недельные продажи в штуках",
		"price": "Историческая недельная цена",
		"promo": "Промо-признак",
	}
	role_map = {
		"series_id": "key",
		"category": "static",
		"week_start": "time_index",
		"sales_units": "target",
		"price": "future_unknown",
		"promo": "future_unknown",
	}

	rows: list[dict[str, object]] = []
	for column in weekly.columns:
		rows.append(
			{
				"column_name": column,
				"dtype": str(weekly[column].dtype),
				"role": role_map.get(column, "derived"),
				"non_null_share": float(weekly[column].notna().mean()),
				"example_value": str(weekly[column].dropna().iloc[0]) if weekly[
					column].notna().any() else "",
				"description": descriptions.get(column),
			}
		)

	return pd.DataFrame(rows)


def build_transformation_summary(weekly: pd.DataFrame) -> pd.DataFrame:
	number_of_weeks = int(weekly["week_start"].nunique()) if not weekly.empty else 0
	number_of_series = int(weekly["series_id"].nunique()) if not weekly.empty else 0

	# sell_prices включены в сырой контекст M5, но их наличие в источнике не означает
	# автоматическую допустимость в forecasting-контуре. Ограничение availability
	# контролируется отдельно через feature policy конкретного эксперимента.
	return pd.DataFrame(
		[
			{
				"step_no": 1,
				"step_name": "daily_source_tables",
				"input_object": "sales_train_* + calendar + sell_prices",
				"output_object": "raw daily context",
				"purpose": "Зафиксировать исходную структуру M5 и границы доступной информации",
				"formula_or_rule": "Используются только колонки дней d_* и календарная таблица для привязки дат",
				"why_needed": "Без календаря невозможно корректно агрегировать дни в недели",
			},
			{
				"step_no": 2,
				"step_name": "daily_to_weekly_aggregation",
				"input_object": "daily sales by d_*",
				"output_object": "weekly panel",
				"purpose": "Построить недельную целевую переменную под forecasting-контур",
				"formula_or_rule": "y_(s,w) = sum_{d in w} sales_(s,d)",
				"why_needed": f"После агрегации получаем {number_of_series} рядов и {number_of_weeks} недель на частоте W-MON",
			},
			{
				"step_no": 3,
				"step_name": "weekly_grid_validation",
				"input_object": "weekly panel",
				"output_object": "validated panel",
				"purpose": "Проверить полноту недельной сетки и длину истории рядов",
				"formula_or_rule": "grid_coverage = observed_weeks / expected_weeks",
				"why_needed": "Без полной сетки лаги и rolling-признаки легко сдвигаются неверно",
			},
			{
				"step_no": 4,
				"step_name": "series_segmentation",
				"input_object": "validated panel",
				"output_object": "segment labels",
				"purpose": "Разделить ряды по стабильности, нулям и проблемности",
				"formula_or_rule": "Используются history_weeks, zero_share и variance-based переменные для диагностики",
				"why_needed": "Сегменты помогают объяснить, почему одна и та же модель ведет себя по-разному на разных рядах",
			},
			{
				"step_no": 5,
				"step_name": "supervised_feature_building",
				"input_object": "validated panel + segments",
				"output_object": "supervised frame",
				"purpose": "Преобразовать временной ряд в табличную выборку для ML-моделей",
				"formula_or_rule": "lag_k(t) = y_(t-k); rolling_mean_w(t) = mean(y_(t-w)..y_(t-1))",
				"why_needed": "ML-модели не видят временную ось напрямую и требуют явных признаков памяти/сезонности",
			},
			{
				"step_no": 6,
				"step_name": "feature_policy_filtering",
				"input_object": "candidate features",
				"output_object": "leakage-safe feature set",
				"purpose": "Оставить только признаки, известные на дату прогноза",
				"why_needed": "Это центральное ограничение постановки и обязательное условие корректного backtesting",
			},
		]
	)


def build_example_feature_snapshots(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
	if weekly.empty:
		return {}

	feature_manifest = FeatureManifest(
		enabled=True,
		feature_set="F4",
		lags=[1, 2, 3, 4, 8, 12, 13, 26, 52],
		rolling_windows=[4, 8, 13, 26],
		use_calendar=True,
		use_category_encodings=True,
		use_price=False,
		use_promo=False,
		use_external=False,
	)
	example_series = _select_example_series_per_category(weekly)
	if example_series.empty:
		return {}
	selected_series_ids = example_series["series_id"].astype(str).tolist()
	selected_weekly = weekly.loc[weekly["series_id"].isin(selected_series_ids)].copy()
	selected_segments = segments.loc[segments["series_id"].isin(selected_series_ids)].copy()
	supervised, _ = make_supervised_frame(selected_weekly, selected_segments, feature_manifest)
	columns = [
		"series_id",
		"category",
		"week_start",
		"sales_units",
		"segment_label",
		"lag_1",
		"lag_13",
		"lag_52",
		"rolling_4_mean",
		"rolling_13_mean",
		"rolling_4_ratio_to_rolling_mean",
		"rolling_13_recent_trend_delta",
		"week_of_year",
		"month",
		"quarter",
		"category_code",
		"segment_code",
	]
	available_columns = [column for column in columns if column in supervised.columns]
	snapshots: dict[str, pd.DataFrame] = {}
	for _, series_row in example_series.iterrows():
		example_series_id = str(series_row["series_id"])
		category = str(series_row["category"])
		snapshot = supervised.loc[
			supervised["series_id"] == example_series_id, available_columns].copy()
		snapshot = snapshot.dropna(subset=["lag_52"], how="any")
		if snapshot.empty:
			snapshot = supervised.loc[
				supervised["series_id"] == example_series_id, available_columns].copy()
		snapshots[category] = snapshot.tail(20).reset_index(drop=True)
	return snapshots


def _select_example_series_per_category(weekly: pd.DataFrame) -> pd.DataFrame:
	return (
		weekly.groupby(["category", "series_id"], as_index=False)
		.agg(total_sales_units=("sales_units", "sum"))
		.sort_values(["category", "total_sales_units", "series_id"], ascending=[True, False, True])
		.drop_duplicates(subset=["category"], keep="first")
		.reset_index(drop=True)
	)


def safe_autocorr(series: pd.Series, lag: int) -> float:
	if len(series) <= lag:
		return float("nan")
	if float(series.std(ddof=0)) == 0.0:
		return 0.0
	return float(series.autocorr(lag=lag))


def _safe_adf(series: pd.Series) -> tuple[float, float]:
	values = series.dropna().astype(float)
	if len(values) < 10 or float(values.std(ddof=0)) == 0.0:
		return float("nan"), float("nan")
	stat, pvalue, *_ = adfuller(values, autolag="AIC")
	return float(stat), float(pvalue)


def _safe_kpss(series: pd.Series) -> tuple[float, float]:
	values = series.dropna().astype(float)
	if len(values) < 10 or float(values.std(ddof=0)) == 0.0:
		return float("nan"), float("nan")
	with warnings.catch_warnings(record=True) as caught_warnings:
		warnings.simplefilter("always", InterpolationWarning)
		stat, pvalue, *_ = kpss(values, regression="c", nlags="auto")
	for warning in caught_warnings:
		if issubclass(warning.category, InterpolationWarning):
			message = str(warning.message)
			if "greater than" in message:
				return float(stat), 0.1
			if "smaller than" in message:
				return float(stat), 0.01
	return float(stat), float(pvalue)


def _infer_stationarity(adf_pvalue: float, kpss_pvalue: float) -> str:
	if np.isnan(adf_pvalue) or np.isnan(kpss_pvalue):
		return "inconclusive"
	if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
		return "likely_stationary"
	if adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
		return "likely_non_stationary"
	return "mixed_signal"


def _compute_gini(values: np.ndarray) -> float:
	if values.size == 0:
		return float("nan")
	sorted_values = np.sort(values)
	if np.allclose(sorted_values.sum(), 0.0):
		return 0.0
	index = np.arange(1, sorted_values.size + 1, dtype=float)
	return float((2 * np.sum(index * sorted_values) / (sorted_values.size * sorted_values.sum())) - ((sorted_values.size + 1) / sorted_values.size))
