from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from mt.domain.manifest import BacktestManifest
from mt.domain.manifest import FeatureManifest
from mt.infra.artifact.plot_labels import set_axis_labels, translate_label
from mt.infra.audit.paths import audit_artifact_relpath, audit_example_series_relpath
from mt.infra.backtest.backtest import build_backtest_windows
from mt.infra.feature.supervised_builder import make_supervised_frame


def save_data_audit_plots(
	summary: pd.DataFrame,
	segments: pd.DataFrame,
	seasonality_summary: pd.DataFrame,
	artifacts_dir: Path,
	aggregation_level: str = "category",
	weekly: pd.DataFrame | None = None,
	raw_context: dict[str, object] | None = None,
) -> None:
	raw_context = raw_context or {}
	artifacts_dir.mkdir(parents=True, exist_ok=True)
	scope_prefix = "Category Audit" if aggregation_level == "category" else "SKU Audit"
	category_context_prefix = (
		"Category Audit"
		if aggregation_level == "category"
		else "Category Context for SKU Audit"
	)
	series_label = "category series" if aggregation_level == "category" else "sku series"

	counts = segments["segment_label"].value_counts().sort_index()
	if counts.shape[0] >= 2:
		fig, ax = plt.subplots(figsize=(8, 4))
		counts.plot(kind="bar", ax=ax, title=f"{scope_prefix}: сегменты рядов ({series_label})")
		set_axis_labels(ax, xlabel="segment", ylabel="count")
		_save_figure(fig, _audit_plot_path(artifacts_dir, "segment_distribution.png"))

	_save_histogram(
		summary,
		"history_weeks",
		f"{scope_prefix}: распределение длины истории",
		_audit_plot_path(artifacts_dir, "history_length_distribution.png")
	)
	_save_histogram(
		summary,
		"zero_share",
		f"{scope_prefix}: распределение доли нулевых продаж",
		_audit_plot_path(artifacts_dir, "zero_share_distribution.png")
	)
	_save_histogram(
		summary,
		"missing_share",
		f"{scope_prefix}: распределение доли пропусков",
		_audit_plot_path(artifacts_dir, "missing_share_distribution.png")
	)
	_save_histogram(
		summary,
		"outlier_share",
		f"{scope_prefix}: распределение доли выбросов",
		_audit_plot_path(artifacts_dir, "outlier_share_distribution.png")
	)
	_save_histogram(
		summary,
		"coefficient_of_variation",
		f"{scope_prefix}: распределение коэффициента вариации",
		_audit_plot_path(artifacts_dir, "coefficient_of_variation_distribution.png")
	)
	_save_histogram(
		summary,
		"trend_strength",
		f"{scope_prefix}: распределение силы тренда",
		_audit_plot_path(artifacts_dir, "trend_strength_distribution.png")
	)

	if summary["history_weeks"].nunique() > 1 and summary["zero_share"].nunique() > 1:
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.scatter(summary["history_weeks"], summary["zero_share"], alpha=0.6)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: доля нулевых продаж и длина истории",
			xlabel="history_weeks",
			ylabel="zero_share",
		)
		_save_figure(fig, _audit_plot_path(artifacts_dir, "zero_share_vs_history.png"))

	if "category" in summary.columns:
		category_counts = summary.groupby("category")["series_id"].nunique().sort_values(
			ascending=False)
		fig, ax = plt.subplots(figsize=(8, 4))
		category_counts.plot(
			kind="bar",
			ax=ax,
			title=f"{category_context_prefix}: число рядов по категориям"
		)
		set_axis_labels(ax, xlabel="category", ylabel="number_of_series")
		_save_figure(fig, _audit_plot_path(artifacts_dir, "series_count_by_category.png"))

	if weekly is not None and not weekly.empty:
		daily_total = raw_context.get("daily_total_sales")
		if isinstance(daily_total, pd.DataFrame) and not daily_total.empty:
			fig, ax = plt.subplots(figsize=(10, 4))
			ax.plot(
				daily_total["date"],
				daily_total["sales_units"],
				linewidth=0.8,
				alpha=0.45,
				label="daily total"
			)
			weekly_total_for_overlay = weekly.groupby("week_start", as_index=False)[
				"sales_units"].sum().copy()
			ax.plot(
				weekly_total_for_overlay["week_start"],
				weekly_total_for_overlay["sales_units"],
				linewidth=1.5,
				label="weekly total"
			)
			set_axis_labels(
				ax,
				title=f"{scope_prefix}: сравнение дневных и недельных суммарных продаж",
				xlabel="date",
				ylabel="sales_units",
			)
			ax.legend(["Дневная сумма", "Недельная сумма"])
			_save_figure(fig,
			             _audit_plot_path(artifacts_dir, "aggregation_daily_vs_weekly_total_sales.png"))

			monthly_total = (
				weekly.assign(month_start=weekly["week_start"].dt.to_period("M").dt.to_timestamp())
				.groupby("month_start", as_index=False)["sales_units"]
				.sum()
			)
			fig, ax = plt.subplots(figsize=(10, 4))
			ax.plot(
				daily_total["date"],
				daily_total["sales_units"],
				linewidth=0.7,
				alpha=0.30,
				label="daily total",
			)
			ax.plot(
				weekly_total_for_overlay["week_start"],
				weekly_total_for_overlay["sales_units"],
				linewidth=1.2,
				label="weekly total",
			)
			ax.plot(
				monthly_total["month_start"],
				monthly_total["sales_units"],
				linewidth=1.4,
				label="monthly total",
			)
			set_axis_labels(
				ax,
				title=f"{scope_prefix}: сравнение дневных, недельных и месячных суммарных продаж",
				xlabel="date",
				ylabel="sales_units",
			)
			ax.legend(["Дневная сумма", "Недельная сумма", "Месячная сумма"])
			_save_figure(fig, _audit_plot_path(artifacts_dir,
			                                   "aggregation_daily_weekly_monthly_total_sales.png"))

		category_sales = weekly.groupby("category", as_index=False)["sales_units"].sum().sort_values(
			"sales_units", ascending=False)
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.bar(category_sales["category"], category_sales["sales_units"])
		set_axis_labels(
			ax,
			title=f"{category_context_prefix}: суммарные продажи по категориям",
			xlabel="category",
			ylabel="sales_units",
		)
		_save_figure(fig, _audit_plot_path(artifacts_dir, "total_sales_by_category.png"))

		weekly_total = weekly.groupby("week_start", as_index=False)["sales_units"].sum()
		fig, ax = plt.subplots(figsize=(10, 4))
		ax.plot(weekly_total["week_start"], weekly_total["sales_units"], linewidth=1.2)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: суммарные недельные продажи",
			xlabel="week_start",
			ylabel="sales_units",
		)
		_save_figure(fig, _audit_plot_path(artifacts_dir, "weekly_total_sales.png"))

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.hist(weekly["sales_units"], bins=25, density=True, alpha=0.75)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: распределение целевой переменной",
			xlabel="sales_units",
			ylabel="density",
		)
		_save_figure(fig, _audit_plot_path(artifacts_dir, "target_distribution_raw.png"))

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.hist(np.log1p(weekly["sales_units"]), bins=25, density=True, alpha=0.75)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: распределение log1p целевой переменной",
			xlabel="log1p(sales_units)",
			ylabel="density",
		)
		_save_figure(fig, _audit_plot_path(artifacts_dir, "target_distribution_log1p.png"))

		category_pivot_source = (
			weekly.groupby(["week_start", "category"], as_index=False)["sales_units"]
			.sum()
		)
		category_pivot = category_pivot_source.pivot(
			index="week_start",
			columns="category",
			values="sales_units",
		).sort_index()
		if not category_pivot.empty:
			normalized = category_pivot / category_pivot.mean(axis=0)
			fig, ax = plt.subplots(figsize=(10, 4))
			for category in normalized.columns:
				ax.plot(normalized.index, normalized[category], linewidth=1.1, label=category)
			set_axis_labels(
				ax,
				title=f"{category_context_prefix}: нормированные недельные продажи по категориям",
				xlabel="week_start",
				ylabel="sales / category_mean",
			)
			ax.legend(title=translate_label("category"))
			_save_figure(fig, _audit_plot_path(artifacts_dir, "normalized_weekly_sales_by_category.png"))

			seasonal_profile = weekly.copy()
			seasonal_profile["week_of_year"] = seasonal_profile[
				"week_start"].dt.isocalendar().week.astype(int)
			seasonal_profile = (
				seasonal_profile.groupby(["category", "week_of_year"], as_index=False)["sales_units"]
				.mean()
				.rename(columns={"sales_units": "mean_sales_units"})
				.sort_values(["category", "week_of_year"])
			)
			category_means = (
				weekly.groupby("category", as_index=False)["sales_units"]
				.mean()
				.rename(columns={"sales_units": "category_mean_sales_units"})
			)
			seasonal_profile = seasonal_profile.merge(category_means, on="category", how="left")
			seasonal_profile["seasonal_index"] = (
				seasonal_profile["mean_sales_units"] / seasonal_profile[
				"category_mean_sales_units"].replace(0, np.nan)
			)
			fig, ax = plt.subplots(figsize=(10, 4))
			for category, frame in seasonal_profile.groupby("category"):
				ax.plot(frame["week_of_year"], frame["seasonal_index"], linewidth=1.1, label=category)
			set_axis_labels(
				ax,
				title=f"{category_context_prefix}: сезонный индекс по неделям года",
				xlabel="week_of_year",
				ylabel="seasonal_index_vs_category_mean",
			)
			ax.legend(title=translate_label("category"))
			_save_figure(fig, _audit_plot_path(artifacts_dir, "seasonal_profile_by_week_of_year.png"))

			seasonal_heatmap = (
				seasonal_profile.pivot(index="category", columns="week_of_year", values="seasonal_index")
				.sort_index()
			)
			if not seasonal_heatmap.empty:
				fig, ax = plt.subplots(figsize=(12, 4))
				image = ax.imshow(seasonal_heatmap.values, aspect="auto", vmin=0.7, vmax=1.3)
				set_axis_labels(
					ax,
					title=f"{category_context_prefix}: тепловая карта сезонного индекса",
					xlabel="week_of_year",
					ylabel="category",
				)
				ax.set_yticks(range(len(seasonal_heatmap.index)), labels=seasonal_heatmap.index)
				x_positions = np.linspace(0, len(seasonal_heatmap.columns) - 1,
				                          num=min(13, len(seasonal_heatmap.columns)), dtype=int)
				ax.set_xticks(x_positions,
				              labels=[int(seasonal_heatmap.columns[pos]) for pos in x_positions])
				fig.colorbar(image, ax=ax, label=translate_label("seasonal_index_vs_category_mean"))
				_save_figure(fig, _audit_plot_path(artifacts_dir, "seasonal_heatmap_by_category_week.png"))

		if not seasonality_summary.empty:
			acf_columns = [
				column for column in
				["acf_lag_1", "acf_lag_4", "acf_lag_8", "acf_lag_13", "acf_lag_26", "acf_lag_52"]
				if column in seasonality_summary.columns
			]
			if acf_columns:
				acf_profile = pd.DataFrame(
					{
						"lag": [int(column.split("_")[-1]) for column in acf_columns],
						"mean_acf": [float(seasonality_summary[column].mean()) for column in acf_columns],
					}
				)
				fig, ax = plt.subplots(figsize=(8, 4))
				ax.bar(acf_profile["lag"].astype(str), acf_profile["mean_acf"])
				set_axis_labels(
					ax,
					title=f"{scope_prefix}: профиль сезонной автокорреляции",
					xlabel="lag_weeks",
					ylabel="mean_acf",
				)
				_save_figure(fig, _audit_plot_path(artifacts_dir, "seasonal_autocorrelation_profile.png"))

			_save_example_series_plots(weekly, segments, artifacts_dir, aggregation_level)
			_save_backtest_schematic(weekly, artifacts_dir, aggregation_level)
			if aggregation_level == "sku":
				_save_sku_aggregation_plots(weekly, artifacts_dir)

	item_counts = raw_context.get("item_counts_by_category")
	if isinstance(item_counts, pd.DataFrame) and not item_counts.empty:
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.bar(item_counts["category"], item_counts["item_count"])
		set_axis_labels(
			ax,
			title=f"{category_context_prefix}: число SKU по категориям",
			xlabel="category",
			ylabel="item_count",
		)
		_save_figure(fig, _audit_plot_path(artifacts_dir, "sku_count_by_category.png"))


def _save_histogram(frame: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(8, 4))
	frame[column].dropna().plot(kind="hist", bins=20, ax=ax, title=title)
	set_axis_labels(ax, xlabel=column, ylabel="count")
	_save_figure(fig, output_path)


def _audit_plot_path(artifacts_dir: Path, filename: str) -> Path:
	return artifacts_dir / audit_artifact_relpath(filename)


def _save_figure(fig: plt.Figure, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def _save_example_series_plots(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
	artifacts_dir: Path,
	aggregation_level: str,
) -> None:
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
	example_series = (
		weekly.groupby(["category", "series_id"], as_index=False)
		.agg(total_sales_units=("sales_units", "sum"))
		.sort_values(["category", "total_sales_units", "series_id"], ascending=[True, False, True])
		.drop_duplicates(subset=["category"], keep="first")
		.reset_index(drop=True)
	)
	if example_series.empty:
		return
	selected_series_ids = example_series["series_id"].astype(str).tolist()
	selected_weekly = weekly.loc[weekly["series_id"].isin(selected_series_ids)].copy()
	selected_segments = segments.loc[segments["series_id"].isin(selected_series_ids)].copy()
	supervised, _ = make_supervised_frame(selected_weekly, selected_segments, feature_manifest)
	for _, series_row in example_series.iterrows():
		example_series_id = str(series_row["series_id"])
		category = str(series_row["category"])
		example = supervised.loc[supervised["series_id"] == example_series_id].sort_values(
			"week_start"
		).copy()
		if example.empty:
			continue
		category_dir = artifacts_dir / audit_example_series_relpath(category, aggregation_level)
		_save_single_example_series_plots(
			example,
			category_dir,
			example_series_id,
			category,
			aggregation_level,
		)


def _save_sku_aggregation_plots(
	weekly: pd.DataFrame,
	artifacts_dir: Path,
) -> None:
	sku_summary = (
		weekly.groupby(["series_id", "category"], as_index=False)
		.agg(
			total_sales_units=("sales_units", "sum"),
			mean_sales_units=("sales_units", "mean"),
		)
		.sort_values(["total_sales_units", "mean_sales_units", "series_id"],
		             ascending=[False, False, True])
		.reset_index(drop=True)
	)
	if sku_summary.empty:
		return

	top_20 = sku_summary.head(20).copy()
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.bar(top_20["series_id"], top_20["total_sales_units"])
	set_axis_labels(
		ax,
		title="SKU Audit: топ-20 SKU по суммарным продажам",
		xlabel="series_id",
		ylabel="total_sales_units",
	)
	ax.tick_params(axis="x", rotation=75)
	_save_figure(fig, _audit_plot_path(artifacts_dir, "sku_top20_total_sales.png"))

	total_sales = float(sku_summary["total_sales_units"].sum())
	if total_sales > 0:
		concentration = sku_summary.loc[:, ["series_id", "total_sales_units"]].copy()
		concentration["rank"] = np.arange(1, len(concentration) + 1)
		concentration["cumulative_sales_share"] = (
			concentration["total_sales_units"].cumsum() / total_sales
		)
		top_limit = min(200, len(concentration))
		fig, ax = plt.subplots(figsize=(10, 4.5))
		ax.plot(
			concentration["rank"].head(top_limit),
			concentration["cumulative_sales_share"].head(top_limit),
			linewidth=1.6,
		)
		set_axis_labels(
			ax,
			title="SKU Audit: накопленная доля продаж по рангу SKU",
			xlabel="sku_rank",
			ylabel="cumulative_sales_share",
		)
		ax.set_ylim(0.0, 1.0)
		ax.grid(alpha=0.25)
		_save_figure(fig, _audit_plot_path(artifacts_dir, "sku_cumulative_sales_share.png"))

	sample_ids = sku_summary["series_id"].head(8).tolist()
	sample_weekly = (
		weekly.loc[weekly["series_id"].isin(sample_ids), ["series_id", "week_start", "sales_units"]]
		.copy()
	)
	if sample_weekly.empty:
		return
	pivot = sample_weekly.pivot(index="week_start", columns="series_id",
	                            values="sales_units").sort_index()
	normalized = pivot.divide(pivot.mean(axis=0), axis=1)
	fig, ax = plt.subplots(figsize=(11, 4.5))
	for series_id in normalized.columns:
		ax.plot(normalized.index, normalized[series_id], linewidth=1.0, label=series_id)
	set_axis_labels(
		ax,
		title="SKU Audit: нормированные недельные продажи для выборки топ-SKU",
		xlabel="week_start",
		ylabel="sales / sku_mean",
	)
	ax.legend(ncol=2, fontsize=8, title=translate_label("series_id"))
	_save_figure(fig, _audit_plot_path(artifacts_dir, "sku_normalized_sample.png"))


def _save_single_example_series_plots(
	example: pd.DataFrame,
	artifacts_dir: Path,
	series_id: str,
	category: str,
	aggregation_level: str,
) -> None:
	scope_prefix = "Category Audit" if aggregation_level == "category" else "SKU Audit"
	fig, ax = plt.subplots(figsize=(11, 4.5))
	ax.plot(example["week_start"], example["sales_units"], linewidth=1.4, label="Продажи, шт.")
	if "lag_1" in example.columns:
		ax.plot(example["week_start"], example["lag_1"], linewidth=1.0, alpha=0.8, label="Лаг 1 нед.")
	if "rolling_4_mean" in example.columns:
		ax.plot(example["week_start"], example["rolling_4_mean"], linewidth=1.2,
		        label="Скользящее среднее 4 нед.")
	if "rolling_13_mean" in example.columns:
		ax.plot(example["week_start"], example["rolling_13_mean"], linewidth=1.2,
		        label="Скользящее среднее 13 нед.")
	set_axis_labels(
		ax,
		title=f"{scope_prefix}: пример ряда и признаков для {series_id} ({category})",
		xlabel="week_start",
		ylabel="sales_units",
	)
	ax.legend()
	_save_figure(fig, artifacts_dir / "example_series_feature_overlay.png")

	zoom = example.tail(40).copy()
	if not zoom.empty:
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.plot(zoom["week_start"], zoom["sales_units"], linewidth=1.4, label="Продажи, шт.")
		if "lag_1" in zoom.columns:
			ax.plot(zoom["week_start"], zoom["lag_1"], linewidth=1.0, alpha=0.8, label="Лаг 1 нед.")
		if "rolling_4_mean" in zoom.columns:
			ax.plot(zoom["week_start"], zoom["rolling_4_mean"], linewidth=1.2,
			        label="Скользящее среднее 4 нед.")
		if "rolling_13_mean" in zoom.columns:
			ax.plot(zoom["week_start"], zoom["rolling_13_mean"], linewidth=1.2,
			        label="Скользящее среднее 13 нед.")
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: пример ряда и признаков на недавнем окне для {series_id} ({category})",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.legend()
		_save_figure(fig, artifacts_dir / "example_series_feature_overlay_recent.png")

	if "lag_52" in example.columns and example["lag_52"].notna().any():
		comparable = example.loc[example["lag_52"].notna()].copy()
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.plot(comparable["week_start"], comparable["sales_units"], linewidth=1.4, label="Факт")
		ax.plot(comparable["week_start"], comparable["lag_52"], linewidth=1.2,
		        label="Seasonal Naive")
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: пример ряда и reference Seasonal Naive для {series_id} ({category})",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.legend()
		_save_figure(fig, artifacts_dir / "example_series_seasonal_naive_overlay.png")

	rolling_smoothed = example[["week_start", "sales_units"]].copy()
	rolling_smoothed["rolling_10"] = (
		rolling_smoothed["sales_units"].rolling(window=10, min_periods=1).mean()
	)
	fig, ax = plt.subplots(figsize=(11, 4.5))
	ax.plot(rolling_smoothed["week_start"], rolling_smoothed["sales_units"], linewidth=0.9,
	        alpha=0.55, label="Продажи, шт.")
	ax.plot(rolling_smoothed["week_start"], rolling_smoothed["rolling_10"], linewidth=1.4,
	        label="Скользящее среднее 10 нед.")
	set_axis_labels(
		ax,
		title=f"{scope_prefix}: сглаживание ряда окном 10 недель для {series_id} ({category})",
		xlabel="week_start",
		ylabel="sales_units",
	)
	ax.legend()
	_save_figure(fig, artifacts_dir / "example_series_smoothed_window_10.png")

	_save_example_acf_pacf(example, artifacts_dir, f"{scope_prefix}: {series_id} ({category})")
	_save_example_decomposition(example, artifacts_dir, f"{scope_prefix}: {series_id} ({category})")


def _save_backtest_schematic(
	weekly: pd.DataFrame,
	artifacts_dir: Path,
	aggregation_level: str,
) -> None:
	if weekly.empty:
		return

	manifest = BacktestManifest(
		horizon_min=1,
		horizon_max=3,
		min_train_weeks=60,
		step_weeks=4,
		max_windows=2,
		mode="direct",
	)

	if weekly["week_start"].nunique() < manifest.min_train_weeks + manifest.horizon_max:
		return

	windows = build_backtest_windows(
		weekly=weekly,
		aggregation_level=aggregation_level,
		manifest=manifest,
		feature_set="F4",
		seed=42,
	)

	if not windows:
		return

	frame = pd.DataFrame(
		{
			"forecast_origin": [window.forecast_origin for window in windows],
			"train_end": [window.train_end for window in windows],
			"test_start": [window.test_start for window in windows],
			"horizon": [window.horizon for window in windows],
		}
	).sort_values(["forecast_origin", "horizon"]).head(6).reset_index(drop=True)
	fig, ax = plt.subplots(figsize=(11, 3.8))
	for idx, row in frame.iterrows():
		ax.plot([row["train_end"], row["forecast_origin"]], [idx, idx], linewidth=6, color="#4C78A8",
		        solid_capstyle="butt")
		ax.plot([row["forecast_origin"], row["test_start"]], [idx, idx], linewidth=3, color="#F58518",
		        linestyle="--")
		ax.scatter([row["forecast_origin"]], [idx], color="#E45756", s=35, zorder=3)
		ax.text(row["test_start"], idx + 0.08, f"h={int(row['horizon'])}", fontsize=8)
	set_axis_labels(ax, title="Схема rolling window backtesting", xlabel="date")
	ax.set_yticks(range(len(frame)), labels=[f"window_{idx + 1}" for idx in range(len(frame))])
	ax.grid(axis="x", alpha=0.25)
	_save_figure(fig, _audit_plot_path(artifacts_dir, "rolling_backtest_schematic.png"))


def _save_example_acf_pacf(example: pd.DataFrame, artifacts_dir: Path, series_id: str) -> None:
	values = example["sales_units"].astype(float).reset_index(drop=True)
	if len(values) < 20:
		return
	fig, axes = plt.subplots(1, 2, figsize=(10, 4))
	max_lag = min(24, len(values) // 2 - 1)
	if max_lag < 2:
		plt.close(fig)
		return
	plot_acf(values, lags=max_lag, ax=axes[0])
	plot_pacf(values, lags=max_lag, ax=axes[1], method="ywm")
	axes[0].set_title("ACF")
	axes[1].set_title("PACF")
	fig.suptitle(f"ACF/PACF для примера ряда: {series_id}")
	_save_figure(fig, artifacts_dir / "example_series_acf_pacf.png")


def _save_example_decomposition(example: pd.DataFrame, artifacts_dir: Path, series_id: str) -> None:
	series = example.set_index("week_start")["sales_units"].astype(float).sort_index()
	if len(series) < 104:
		return
	result = seasonal_decompose(series, model="additive", period=52, extrapolate_trend="freq")
	fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
	series.plot(ax=axes[0], title=f"Декомпозиция временного ряда: {series_id}")
	result.trend.plot(ax=axes[1], title="Тренд")
	result.seasonal.plot(ax=axes[2], title="Сезонная компонента")
	result.resid.plot(ax=axes[3], title="Остаток")
	for ax in axes:
		ax.grid(alpha=0.2)
	_save_figure(fig, artifacts_dir / "example_series_decomposition.png")
