from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt

from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.forecast.forecast_artifact import ForecastArtifactPathsMap
from mt.infra.artifact.plot_labels import set_axis_labels
from mt.infra.artifact.plot_writer import save_figure


def write_overlay_plots(ctx: ForecastPipelineContext) -> None:
	if ctx.frame is None or ctx.predictions is None:
		raise ValueError()

	frame = ctx.frame
	predictions = ctx.predictions
	for overlay_prediction in _iter_series_predictions(predictions):
		series_id = str(overlay_prediction["series_id"].iloc[0])
		series_frame = frame.loc[frame["series_id"] == series_id].sort_values("week_start").reset_index(
			drop=True)
		history = series_frame.loc[series_frame["is_history"].astype(bool)]
		if history.empty:
			continue
		zoom_history_weeks = _resolve_zoom_history_weeks(history)
		plot_history_weeks = _resolve_plot_history_weeks(history, len(overlay_prediction))
		future = series_frame.loc[~series_frame["is_history"].astype(bool)]
		forecast_origin = pd.Timestamp(overlay_prediction["forecast_origin"].iloc[0])
		output_path = _series_plot_path(ctx.artifacts_paths_map, "forecast_overlay", series_id)

		fig, (ax_full, ax_zoom) = plt.subplots(
			2,
			1,
			figsize=(16, 9.5),
			sharex=False,
			height_ratios=[1.15, 1.0],
		)
		visible_future = future.head(len(overlay_prediction))
		plot_history = history.tail(plot_history_weeks)
		zoom_history = history.tail(zoom_history_weeks)
		_plot_overlay_axis(
			ax_full,
			plot_history,
			visible_future,
			overlay_prediction,
			forecast_origin,
			f"Forecast и факт: {series_id}",
		)
		_plot_overlay_axis(
			ax_zoom,
			zoom_history,
			visible_future,
			overlay_prediction,
			forecast_origin,
			f"Крупный план: последние {zoom_history_weeks} недель истории и окно прогноза",
		)
		save_figure(fig, output_path, dpi=160)


def write_actual_vs_prediction_plot(predictions: pd.DataFrame, output_dir: Path) -> None:
	scored_predictions = predictions.loc[predictions["actual"].notna()].copy()
	if scored_predictions.empty:
		return
	output_path = output_dir / "forecast_actual_vs_prediction.png"
	fig, ax = plt.subplots(figsize=(9.5, 7.5))
	ax.scatter(
		scored_predictions["actual"],
		scored_predictions["prediction"],
		alpha=0.55,
		s=38,
		color="#1f77b4",
	)
	all_values = pd.concat(
		[
			scored_predictions["actual"].astype(float),
			scored_predictions["prediction"].astype(float),
		],
		ignore_index=True,
	)
	if all_values.empty:
		min_value = 0.0
		max_value = 1.0
	else:
		min_value = float(all_values.min())
		max_value = float(all_values.max())
	padding = max((max_value - min_value) * 0.08, 1.0)
	lower = min(0.0, min_value - padding)
	upper = max_value + padding
	ax.plot([lower, upper], [lower, upper], linestyle="--", color="#5c677d", linewidth=1.5)
	ax.set_xlim(lower, upper)
	ax.set_ylim(lower, upper)
	set_axis_labels(ax, title="Факт против прогноза по всем forecast points", xlabel="actual",
	                ylabel="prediction")
	ax.grid(alpha=0.2, linestyle="--")
	save_figure(fig, output_path, dpi=160)


def write_abs_error_by_horizon_plot(predictions: pd.DataFrame, output_dir: Path) -> None:
	output_path = output_dir / "forecast_abs_error_by_horizon.png"
	frame = predictions.copy()
	frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()
	pivot = frame.groupby("horizon", as_index=False)["abs_error"].mean().sort_values("horizon")
	fig, ax = plt.subplots(figsize=(10.5, 6.8))
	ax.plot(pivot["horizon"], pivot["abs_error"], linewidth=2.6, marker="o", markersize=6)
	set_axis_labels(ax, title="Средняя абсолютная ошибка по горизонтам", xlabel="horizon",
	                ylabel="mean absolute error")
	ax.set_xticks(sorted(pivot["horizon"].unique().tolist()))
	ax.grid(alpha=0.25, linestyle="--")
	save_figure(fig, output_path, dpi=160)


def write_wape_by_horizon_plot(metrics: pd.DataFrame, output_dir: Path) -> None:
	output_path = output_dir / "forecast_wape_by_horizon.png"
	if metrics.empty:
		return
	fig, ax = plt.subplots(figsize=(10.5, 6.8))
	ax.plot(metrics["horizon"], metrics["WAPE"], linewidth=2.8, marker="o", markersize=6)
	set_axis_labels(ax, title="WAPE по горизонтам", xlabel="horizon", ylabel="WAPE")
	ax.set_xticks(sorted(metrics["horizon"].unique().tolist()))
	ax.grid(alpha=0.25, linestyle="--")
	save_figure(fig, output_path, dpi=160)


def _plot_overlay_axis(
	ax: plt.Axes,
	history_frame: pd.DataFrame,
	future_frame: pd.DataFrame,
	overlay_prediction: pd.DataFrame,
	forecast_origin: pd.Timestamp,
	title: str,
) -> None:
	actual_line = _prepend_history_anchor(
		history_frame=history_frame,
		target_frame=future_frame,
		date_column="week_start",
		value_column="sales_units",
	)
	forecast_line = _prepend_forecast_anchor(history_frame, overlay_prediction)
	ax.plot(history_frame["week_start"], history_frame["sales_units"], label="История продаж",
	        color="#1f4e79", linewidth=2.2)
	ax.plot(actual_line["week_start"], actual_line["sales_units"], label="Факт после даты прогноза",
	        color="#2a9d8f", linewidth=2.2, marker="o", markersize=4)
	ax.plot(forecast_line["target_date"], forecast_line["prediction"], label=f"Прогноз модели",
	        color="#d1495b", linewidth=2.2, marker="o")
	if overlay_prediction["lo_95"].notna().any() and overlay_prediction["hi_95"].notna().any():
		ax.fill_between(overlay_prediction["target_date"], overlay_prediction["lo_95"],
		                overlay_prediction["hi_95"], color="#d1495b", alpha=0.10,
		                label="95% прогнозный интервал")
	if overlay_prediction["lo_80"].notna().any() and overlay_prediction["hi_80"].notna().any():
		ax.fill_between(overlay_prediction["target_date"], overlay_prediction["lo_80"],
		                overlay_prediction["hi_80"], color="#d1495b", alpha=0.18,
		                label="80% прогнозный интервал")
	ax.axvline(forecast_origin, color="#5c677d", linestyle="--", linewidth=1.4,
	           label="Начало прогноза")
	set_axis_labels(ax, title=title, xlabel="week_start", ylabel="sales_units")
	ax.legend(loc="upper left", ncols=2, fontsize=8)
	local_values = pd.concat(
		[
			history_frame["sales_units"].astype(float),
			future_frame["sales_units"].astype(float),
			overlay_prediction["prediction"].astype(float),
			overlay_prediction["lo_95"].dropna().astype(float),
			overlay_prediction["hi_95"].dropna().astype(float),
			overlay_prediction["lo_80"].dropna().astype(float),
			overlay_prediction["hi_80"].dropna().astype(float),
		],
		ignore_index=True,
	)
	if not local_values.empty:
		min_value = float(local_values.min())
		max_value = float(local_values.max())
		padding = max((max_value - min_value) * 0.12, 8.0)
		ax.set_ylim(max(0.0, min_value - padding), max_value + padding)
	ax.grid(alpha=0.22, linestyle="--")
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
	for label in ax.get_xticklabels():
		label.set_rotation(25)
		label.set_ha("right")


def write_fan_chart_plots(ctx: ForecastPipelineContext) -> None:
	if ctx.predictions is None or ctx.predictions.empty:
		return
	for overlay_prediction in _iter_series_predictions(ctx.predictions):
		overlay_prediction = overlay_prediction.sort_values("target_date")
		series_id = str(overlay_prediction["series_id"].iloc[0])
		output_path = _series_plot_path(ctx.artifacts_paths_map, "forecast_fan_chart", series_id)
		fig, ax = plt.subplots(figsize=(11.5, 6.4))
		ax.plot(
			overlay_prediction["target_date"],
			overlay_prediction["q50"].fillna(overlay_prediction["prediction"]),
			color="#d1495b",
			linewidth=2.8,
			marker="o",
			label="median forecast",
		)
		if overlay_prediction["lo_95"].notna().any():
			ax.fill_between(
				overlay_prediction["target_date"],
				overlay_prediction["lo_95"],
				overlay_prediction["hi_95"],
				color="#d1495b",
				alpha=0.10,
				label="95% interval",
			)
		if overlay_prediction["lo_80"].notna().any():
			ax.fill_between(
				overlay_prediction["target_date"],
				overlay_prediction["lo_80"],
				overlay_prediction["hi_80"],
				color="#d1495b",
				alpha=0.20,
				label="80% interval",
			)
		ax.plot(
			overlay_prediction["target_date"],
			overlay_prediction["actual"],
			color="#2a9d8f",
			linewidth=2.1,
			marker="o",
			label="actual",
		)
		set_axis_labels(
			ax,
			title=f"Forecast fan chart: {series_id}",
			xlabel="week_start",
			ylabel="sales_units",
		)
		ax.grid(alpha=0.22, linestyle="--")
		ax.legend(loc="upper left")
		save_figure(fig, output_path, dpi=160)


def _prepend_history_anchor(
	history_frame: pd.DataFrame,
	target_frame: pd.DataFrame,
	date_column: str,
	value_column: str,
) -> pd.DataFrame:
	if history_frame.empty or target_frame.empty:
		return target_frame
	last_history = history_frame.sort_values(date_column).iloc[-1]
	anchor = pd.DataFrame(
		[
			{
				date_column: last_history[date_column],
				value_column: last_history[value_column],
			}
		]
	)
	return pd.concat([anchor, target_frame.loc[:, [date_column, value_column]]], ignore_index=True)


def _prepend_forecast_anchor(
	history_frame: pd.DataFrame,
	overlay_prediction: pd.DataFrame,
) -> pd.DataFrame:
	if history_frame.empty or overlay_prediction.empty:
		return overlay_prediction
	last_history = history_frame.sort_values("week_start").iloc[-1]
	anchor = pd.DataFrame(
		[
			{
				"target_date": last_history["week_start"],
				"prediction": last_history["sales_units"],
			}
		]
	)
	return pd.concat(
		[anchor, overlay_prediction.loc[:, ["target_date", "prediction"]]],
		ignore_index=True,
	)


def _iter_series_predictions(predictions: pd.DataFrame) -> list[pd.DataFrame]:
	if predictions.empty:
		return []
	return [
		group.sort_values(["target_date", "horizon"]).reset_index(drop=True)
		for _, group in predictions.groupby("series_id", sort=True)
	]


def _resolve_plot_history_weeks(history: pd.DataFrame, horizon_points: int) -> int:
	recommended = max(16, min(52, horizon_points * 4))
	return min(len(history), recommended)


def _resolve_zoom_history_weeks(history: pd.DataFrame) -> int:
	recommended = max(8, min(20, len(history) // 3 if len(history) >= 24 else len(history)))
	return min(len(history), recommended)


def _series_plot_path(
	artifacts_paths_map: ForecastArtifactPathsMap,
	prefix: str,
	series_id: str,
) -> Path:
	safe_id = "".join(
		character if character.isalnum() else "_" for character in series_id.lower()).strip("_")
	return artifacts_paths_map.plot_file(f"{prefix}__{safe_id or 'unknown'}.png")
