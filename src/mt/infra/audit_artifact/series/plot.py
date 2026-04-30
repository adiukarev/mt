from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.domain.audit.audit_series import compute_outlier_mask
from mt.infra.artifact.plot_labels import set_axis_labels
from mt.infra.artifact.plot_writer import save_figure
from mt.infra.audit.feature_builder import default_feature_manifest
from mt.infra.feature.supervised_builder import build_supervised_frame


def write_series_feature_overlay_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		scope_prefix = _scope_prefix(aggregation_level)
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.plot(series_frame["week_start"], series_frame["sales_units"], linewidth=1.4, label="Продажи, шт.")
		if "lag_1" in series_frame.columns:
			ax.plot(series_frame["week_start"], series_frame["lag_1"], linewidth=1.0, alpha=0.8, label="Лаг 1 нед.")
		if "rolling_mean_7" in series_frame.columns:
			ax.plot(
				series_frame["week_start"],
				series_frame["rolling_mean_7"],
				linewidth=1.2,
				label="Скользящее среднее 7 нед.",
			)
		if "rolling_mean_28" in series_frame.columns:
			ax.plot(
				series_frame["week_start"],
				series_frame["rolling_mean_28"],
				linewidth=1.2,
				label="Скользящее среднее 28 нед.",
			)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: ряд и признаки для {series_id} ({category})",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.legend()
		save_figure(fig, artifacts_dir / "series_feature_overlay.png")


def write_series_feature_overlay_recent_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		zoom = series_frame.tail(40).copy()
		if zoom.empty:
			continue

		scope_prefix = _scope_prefix(aggregation_level)
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.plot(zoom["week_start"], zoom["sales_units"], linewidth=1.4, label="Продажи, шт.")
		if "lag_1" in zoom.columns:
			ax.plot(zoom["week_start"], zoom["lag_1"], linewidth=1.0, alpha=0.8, label="Лаг 1 нед.")
		if "rolling_mean_7" in zoom.columns:
			ax.plot(zoom["week_start"], zoom["rolling_mean_7"], linewidth=1.2, label="Скользящее среднее 7 нед.")
		if "rolling_mean_28" in zoom.columns:
			ax.plot(
				zoom["week_start"],
				zoom["rolling_mean_28"],
				linewidth=1.2,
				label="Скользящее среднее 28 нед.",
			)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: ряд и признаки на недавнем окне для {series_id} ({category})",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.legend()
		save_figure(fig, artifacts_dir / "series_feature_overlay_recent.png")


def write_series_seasonal_naive_overlay_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		comparable = series_frame.copy()
		comparable["seasonal_naive_reference"] = comparable["sales_units"].shift(52)
		if not comparable["seasonal_naive_reference"].notna().any():
			continue

		comparable = comparable.loc[comparable["seasonal_naive_reference"].notna()].copy()
		scope_prefix = _scope_prefix(aggregation_level)
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.plot(comparable["week_start"], comparable["sales_units"], linewidth=1.4, label="Факт, шт./нед.")
		ax.plot(
			comparable["week_start"],
			comparable["seasonal_naive_reference"],
			linewidth=1.2,
			label="Seasonal Naive, шт./нед.",
		)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: ряд и reference Seasonal Naive для {series_id} ({category})",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.legend()
		save_figure(fig, artifacts_dir / "series_seasonal_naive_overlay.png")


def write_series_outlier_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		plot_frame = series_frame.loc[:, ["week_start", "sales_units", "days_in_week"]].copy()
		if plot_frame.empty:
			continue
		complete_mask = plot_frame["days_in_week"].eq(7) if "days_in_week" in plot_frame.columns else pd.Series(
			True,
			index=plot_frame.index,
		)
		complete_values = plot_frame.loc[complete_mask, "sales_units"].astype(float).reset_index(drop=True)
		outlier_mask = compute_outlier_mask(complete_values)
		outlier_points = plot_frame.loc[complete_mask].reset_index(drop=True).loc[outlier_mask.values].copy()

		scope_prefix = _scope_prefix(aggregation_level)
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.plot(
			plot_frame["week_start"],
			plot_frame["sales_units"],
			linewidth=1.3,
			label="Продажи, шт./нед.",
		)
		if not outlier_points.empty:
			ax.scatter(
				outlier_points["week_start"],
				outlier_points["sales_units"],
				color="crimson",
				s=42,
				zorder=3,
				label="Выбросы по MAD",
			)
		incomplete_weeks = plot_frame.loc[~complete_mask]
		if not incomplete_weeks.empty:
			for week_start in incomplete_weeks["week_start"]:
				ax.axvspan(
					pd.Timestamp(week_start) - pd.Timedelta(days=3.5),
					pd.Timestamp(week_start) + pd.Timedelta(days=3.5),
					color="gray",
					alpha=0.15,
				)
			ax.plot([], [], color="gray", alpha=0.35, linewidth=6, label="Неполные недели вне outlier-детекции")
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: выбросы ряда по MAD для {series_id} ({category})",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.legend()
		save_figure(fig, artifacts_dir / "series_outlier_plot.png")


def write_series_smoothed_window_10_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		rolling_smoothed = series_frame[["week_start", "sales_units"]].copy()
		rolling_smoothed["rolling_10"] = rolling_smoothed["sales_units"].rolling(window=10, min_periods=1).mean()
		scope_prefix = _scope_prefix(aggregation_level)
		fig, ax = plt.subplots(figsize=(11, 4.5))
		ax.plot(
			rolling_smoothed["week_start"],
			rolling_smoothed["sales_units"],
			linewidth=0.9,
			alpha=0.55,
			label="Продажи, шт.",
		)
		ax.plot(
			rolling_smoothed["week_start"],
			rolling_smoothed["rolling_10"],
			linewidth=1.4,
			label="Скользящее среднее 10 нед.",
		)
		set_axis_labels(
			ax,
			title=f"{scope_prefix}: сглаживание ряда окном 10 недель для {series_id} ({category})",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.legend()
		save_figure(fig, artifacts_dir / "series_smoothed_window_10.png")


def write_series_acf_pacf_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		values = series_frame["sales_units"].astype(float).reset_index(drop=True)
		if len(values) < 20:
			continue

		fig, axes = plt.subplots(1, 2, figsize=(10, 4))
		max_lag = min(24, len(values) // 2 - 1)
		if max_lag < 2:
			plt.close(fig)
			continue

		plot_acf(values, lags=max_lag, ax=axes[0])
		plot_pacf(values, lags=max_lag, ax=axes[1], method="ywm")
		axes[0].set_title("ACF")
		axes[1].set_title("PACF")
		fig.suptitle(f"ACF/PACF для ряда: {_scope_prefix(aggregation_level)}: {series_id} ({category})")
		save_figure(fig, artifacts_dir / "series_acf_pacf.png")


def write_series_decomposition_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, _aggregation_level in _iter_series_frames(ctx):
		series = series_frame.set_index("week_start")["sales_units"].astype(float).sort_index()
		if len(series) < 104:
			continue

		result = seasonal_decompose(series, model="additive", period=52, extrapolate_trend="freq")
		fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
		series.plot(ax=axes[0], title=f"Декомпозиция временного ряда: {series_id} ({category})")
		result.trend.plot(ax=axes[1], title="Тренд")
		result.seasonal.plot(ax=axes[2], title="Сезонная компонента")
		result.resid.plot(ax=axes[3], title="Остаток")
		for axis in axes:
			axis.grid(alpha=0.2)
		save_figure(fig, artifacts_dir / "series_decomposition.png")


def write_series_noise_comparison_plots(ctx: AuditPipelineContext) -> None:
	for _, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		granularity_frames = _build_noise_granularity_frames(ctx, series_id)
		if len(granularity_frames) < 2:
			continue

		scope_prefix = _scope_prefix(aggregation_level)
		fig, (ax_trend, ax_summary) = plt.subplots(
			2,
			1,
			figsize=(11.5, 8.2),
			height_ratios=[1.8, 1.0],
		)
		summary_rows: list[dict[str, float | str]] = []
		for granularity, frame in granularity_frames.items():
			if frame.empty:
				continue
			ax_trend.plot(
				frame["date"],
				frame["noise_score"],
				linewidth=2.0,
				marker="o",
				markersize=3.5,
				label=granularity,
			)
			summary_rows.append(
				{
					"granularity": granularity,
					"mean_noise": float(frame["noise_score"].mean()),
					"median_noise": float(frame["noise_score"].median()),
				}
			)
		if not summary_rows:
			plt.close(fig)
			continue
		summary = pd.DataFrame(summary_rows).sort_values("granularity").reset_index(drop=True)
		ax_summary.bar(summary["granularity"], summary["mean_noise"], color=["#457b9d", "#2a9d8f", "#e76f51"][:len(summary)])
		for row in summary.itertuples(index=False):
			ax_summary.text(
				row.granularity,
				row.mean_noise,
				f"mean={row.mean_noise:.3f}\nmedian={row.median_noise:.3f}",
				ha="center",
				va="bottom",
				fontsize=9,
			)
		set_axis_labels(
			ax_trend,
			title=f"{scope_prefix}: шумность по дням / неделям / месяцам для {series_id} ({category})",
			xlabel="date",
			ylabel="rolling CV",
		)
		set_axis_labels(
			ax_summary,
			title="Средняя шумность по гранулярностям",
			xlabel="granularity",
			ylabel="mean rolling CV",
		)
		ax_trend.grid(alpha=0.22, linestyle="--")
		ax_summary.grid(alpha=0.18, linestyle="--", axis="y")
		ax_trend.legend(loc="upper left")
		save_figure(fig, artifacts_dir / "series_noise_by_granularity.png")


def write_series_aggregation_granularity_plots(ctx: AuditPipelineContext) -> None:
	for _, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		daily = _build_daily_series_frame(ctx, series_id)
		if daily.empty:
			continue

		weekly = _resample_noise_frame(daily, "W-MON")
		monthly = _resample_noise_frame(daily, "MS")
		if daily.empty and weekly.empty and monthly.empty:
			continue

		scope_prefix = _scope_prefix(aggregation_level)
		fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.8), sharex=False)
		frames = (
			("Дневная агрегация", daily, "#4ea8de"),
			("Недельная агрегация", weekly, "#ffb703"),
			("Месячная агрегация", monthly, "#52b788"),
		)
		for axis, (title, frame, color) in zip(axes, frames, strict=False):
			if frame.empty:
				axis.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center", transform=axis.transAxes)
				axis.set_title(title)
				axis.grid(alpha=0.18, linestyle="--")
				continue
			axis.plot(frame["date"], frame["sales_units"], color=color, linewidth=1.3)
			set_axis_labels(
				axis,
				title=title,
				xlabel="date",
				ylabel="sales_units",
			)
			axis.grid(alpha=0.18, linestyle="--")
		fig.suptitle(f"{scope_prefix}: агрегация ряда по дням / неделям / месяцам для {series_id} ({category})")
		save_figure(fig, artifacts_dir / "series_aggregation_by_granularity.png")


def write_series_sales_distribution_plots(ctx: AuditPipelineContext) -> None:
	for series_frame, artifacts_dir, series_id, category, aggregation_level in _iter_series_frames(ctx):
		values = series_frame["sales_units"].dropna().astype(float).reset_index(drop=True)
		if values.empty:
			continue

		log_values = np.log1p(values.clip(lower=0.0))
		scope_prefix = _scope_prefix(aggregation_level)
		fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2))
		_plot_distribution(
			axes[0],
			values,
			title="Распределение weekly sales_units",
			xlabel="sales_units",
			color="#6baed6",
		)
		_plot_distribution(
			axes[1],
			log_values,
			title="Распределение log1p(sales_units)",
			xlabel="log1p(sales_units)",
			color="#3182bd",
		)
		fig.suptitle(f"{scope_prefix}: распределение продаж для {series_id} ({category})")
		save_figure(fig, artifacts_dir / "series_sales_distribution.png")


def _build_noise_granularity_frames(
	ctx: AuditPipelineContext,
	series_id: str,
) -> dict[str, pd.DataFrame]:
	daily = _build_daily_series_frame(ctx, series_id)
	result: dict[str, pd.DataFrame] = {}
	for granularity, frame in (
		("daily", daily),
		("weekly", _resample_noise_frame(daily, "W-MON")),
		("monthly", _resample_noise_frame(daily, "MS")),
	):
		noise_frame = _build_noise_frame(frame, granularity)
		if not noise_frame.empty:
			result[granularity] = noise_frame
	return result


def _build_noise_frame(frame: pd.DataFrame, granularity: str) -> pd.DataFrame:
	if frame.empty:
		return pd.DataFrame()
	windows = {
		"daily": 28,
		"weekly": 8,
		"monthly": 6,
	}
	window = min(windows[granularity], len(frame))
	if window < 3:
		return pd.DataFrame()
	result = frame.copy()
	rolling_mean = result["sales_units"].rolling(window=window, min_periods=max(3, window // 2)).mean()
	rolling_std = result["sales_units"].rolling(window=window, min_periods=max(3, window // 2)).std(ddof=0)
	result["noise_score"] = (rolling_std / rolling_mean.abs().clip(lower=1e-9)).replace([float("inf")], pd.NA)
	result = result.dropna(subset=["noise_score"]).reset_index(drop=True)
	return result.loc[:, ["date", "noise_score"]]


def _resample_noise_frame(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
	if frame.empty:
		return pd.DataFrame(columns=["date", "sales_units"])
	resampled = (
		frame.set_index("date")["sales_units"]
		.resample(frequency)
		.sum()
		.reset_index()
	)
	return resampled.loc[:, ["date", "sales_units"]]


def _build_daily_series_frame(
	ctx: AuditPipelineContext,
	series_id: str,
) -> pd.DataFrame:
	if ctx.raw_dataset is None:
		return pd.DataFrame(columns=["date", "sales_units"])
	if ctx.raw_dataset.kind.value == "m5":
		return _build_m5_daily_series_frame(ctx, series_id)
	if ctx.raw_dataset.kind.value == "favorita":
		return _build_favorita_daily_series_frame(ctx, series_id)
	if ctx.dataset is None:
		return pd.DataFrame(columns=["date", "sales_units"])
	weekly = ctx.dataset.weekly.loc[ctx.dataset.weekly["series_id"].astype(str) == str(series_id), ["week_start", "sales_units"]].copy()
	if weekly.empty:
		return pd.DataFrame(columns=["date", "sales_units"])
	return weekly.rename(columns={"week_start": "date"}).reset_index(drop=True)


def _build_m5_daily_series_frame(
	ctx: AuditPipelineContext,
	series_id: str,
) -> pd.DataFrame:
	sales = ctx.require_raw_dataset().require_table("sales").copy()
	calendar = ctx.require_raw_dataset().require_table("calendar").copy()
	calendar["date"] = pd.to_datetime(calendar["date"], utc=False)
	day_columns = [column for column in sales.columns if column.startswith("d_")]
	if ctx.dataset is None:
		return pd.DataFrame(columns=["date", "sales_units"])
	if ctx.dataset.aggregation_level == "category":
		selected = sales.loc[sales["cat_id"].astype(str) == str(series_id), day_columns]
		category = str(series_id)
	else:
		selected_rows = sales.loc[sales["item_id"].astype(str) == str(series_id), ["cat_id", *day_columns]].copy()
		if selected_rows.empty:
			return pd.DataFrame(columns=["date", "sales_units"])
		category = str(selected_rows["cat_id"].iloc[0])
		selected = selected_rows.loc[:, day_columns]
	if selected.empty:
		return pd.DataFrame(columns=["date", "sales_units"])
	daily = (
		selected.sum(axis=0)
		.rename_axis("d")
		.reset_index(name="sales_units")
		.merge(calendar.loc[:, ["d", "date"]], on="d", how="left")
		.sort_values("date")
		.reset_index(drop=True)
	)
	return daily.loc[:, ["date", "sales_units"]].assign(series_id=str(series_id), category=category)


def _build_favorita_daily_series_frame(
	ctx: AuditPipelineContext,
	series_id: str,
) -> pd.DataFrame:
	daily_series_by_series_id = ctx.raw_context.get("daily_series_by_series_id")
	if isinstance(daily_series_by_series_id, dict):
		daily = daily_series_by_series_id.get(str(series_id))
		if isinstance(daily, pd.DataFrame) and not daily.empty:
			return daily.loc[:, ["date", "sales_units"]].copy()

	if ctx.dataset is None:
		return pd.DataFrame(columns=["date", "sales_units"])
	weekly = ctx.dataset.weekly.loc[
		ctx.dataset.weekly["series_id"].astype(str) == str(series_id),
		["week_start", "sales_units"],
	].copy()
	if weekly.empty:
		return pd.DataFrame(columns=["date", "sales_units"])
	return weekly.rename(columns={"week_start": "date"}).reset_index(drop=True)


def _iter_series_frames(
	ctx: AuditPipelineContext,
) -> list[tuple[pd.DataFrame, Path, str, str, str]]:
	if ctx.dataset is None or ctx.segments is None:
		raise ValueError()

	weekly = ctx.dataset.weekly
	segments = ctx.segments
	aggregation_level = ctx.dataset.aggregation_level
	selected_series_ids = (
		weekly.loc[:, "series_id"]
		.astype(str)
		.drop_duplicates()
		.sort_values()
		.tolist()
	)
	if not selected_series_ids:
		return []

	selected_weekly = weekly.loc[weekly["series_id"].isin(selected_series_ids)].copy()
	selected_segments = segments.loc[segments["series_id"].isin(selected_series_ids)].copy()
	supervised, _ = build_supervised_frame(selected_weekly, selected_segments, default_feature_manifest())

	result: list[tuple[pd.DataFrame, Path, str, str, str]] = []
	category_lookup = (
		weekly.loc[:, ["series_id", "category"]]
		.drop_duplicates(subset=["series_id"])
		.assign(series_id=lambda frame: frame["series_id"].astype(str))
		.set_index("series_id")["category"]
		.to_dict()
	)
	for series_id in selected_series_ids:
		category = str(category_lookup.get(series_id, "unknown"))
		series_frame = supervised.loc[supervised["series_id"].astype(str) == series_id].sort_values("week_start").copy()
		if series_frame.empty:
			continue
		artifacts_dir = ctx.artifacts_paths_map.series_dir(series_id)
		result.append((series_frame, artifacts_dir, series_id, category, aggregation_level))
	return result


def _scope_prefix(aggregation_level: str) -> str:
	return "Category Audit" if aggregation_level == "category" else "SKU Audit"


def _plot_distribution(
	ax: plt.Axes,
	values: pd.Series,
	*,
	title: str,
	xlabel: str,
	color: str,
) -> None:
	if values.empty:
		ax.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center", transform=ax.transAxes)
		set_axis_labels(ax, title=title, xlabel=xlabel, ylabel="density")
		return

	bins = min(24, max(8, int(np.sqrt(len(values)))))
	ax.hist(values, bins=bins, density=True, color=color, alpha=0.55, edgecolor="white", linewidth=0.8)
	if values.nunique() > 1:
		values.plot(kind="kde", ax=ax, color=color, linewidth=1.5)
	set_axis_labels(ax, title=title, xlabel=xlabel, ylabel="density")
	ax.grid(alpha=0.15, linestyle="--", axis="y")
