from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from mt.infra.artifact.plot_labels import set_axis_labels

ALLOWED_PREDICT_MODELS = {"naive", "seasonal_naive", "moving_average", "auto"}


@dataclass(slots=True)
class SyntheticPredictArtifacts:
	"""Артефакты synthetic inference"""

	predictions_csv: Path
	metrics_csv: Path
	overlay_png: Path
	overlay_zoom_png: Path
	report_md: Path
	pipeline_report_md: Path
	manifest_snapshot_yaml: Path


def write_overlay_plot(
	frame: pd.DataFrame,
	predictions: pd.DataFrame,
	overlay_series_id: str | None,
	plot_history_weeks: int,
	zoom_history_weeks: int,
	annotate_forecast_values: bool,
	output_path: Path,
) -> None:
	overlay_prediction = _resolve_overlay_predictions(predictions, overlay_series_id)
	scenario_name = str(overlay_prediction["scenario_name"].iloc[0])
	series_id = str(overlay_prediction["series_id"].iloc[0])

	mask = frame["series_id"] == series_id
	if "scenario_name" in frame.columns:
		mask = mask & (frame["scenario_name"] == scenario_name)
	series_frame = frame.loc[mask].sort_values("week_start").reset_index(drop=True)

	history = series_frame.loc[series_frame["is_history"].astype(bool)].tail(plot_history_weeks)
	zoom_history = series_frame.loc[series_frame["is_history"].astype(bool)].tail(
		zoom_history_weeks)
	future = series_frame.loc[~series_frame["is_history"].astype(bool)]
	forecast_origin = pd.Timestamp(overlay_prediction["forecast_origin"].iloc[0])

	fig, (ax_full, ax_zoom, ax_forecast_only) = plt.subplots(
		3,
		1,
		figsize=(16, 13),
		sharex=False,
		height_ratios=[1.2, 1.0, 0.95],
	)
	_full_future = future.head(len(overlay_prediction))
	_plot_overlay_axis(
		ax=ax_full,
		history_frame=history,
		future_frame=_full_future,
		overlay_prediction=overlay_prediction,
		forecast_origin=forecast_origin,
		title=f"Синтетический прогноз и факт: {scenario_name} / {series_id}",
		annotate_forecast_values=False,
	)
	_plot_overlay_axis(
		ax=ax_zoom,
		history_frame=zoom_history,
		future_frame=_full_future,
		overlay_prediction=overlay_prediction,
		forecast_origin=forecast_origin,
		title=f"Крупный план: последние {zoom_history_weeks} недель истории и окно прогноза",
		annotate_forecast_values=annotate_forecast_values,
	)
	_plot_forecast_window_axis(
		ax=ax_forecast_only,
		history_frame=zoom_history.tail(min(8, len(zoom_history))),
		future_frame=_full_future,
		overlay_prediction=overlay_prediction,
		forecast_origin=forecast_origin,
		annotate_forecast_values=annotate_forecast_values,
	)
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def write_diagnostic_plots(
	predictions: pd.DataFrame,
	metrics: pd.DataFrame,
	output_dir: Path,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	_write_actual_vs_prediction_plot(predictions, output_dir / "forecast_actual_vs_prediction.png")
	_write_abs_error_by_horizon_plot(predictions, output_dir / "forecast_abs_error_by_horizon.png")
	_write_scenario_wape_plot(metrics, output_dir / "forecast_wape_by_scenario_horizon.png")


def _plot_overlay_axis(
	ax: plt.Axes,
	history_frame: pd.DataFrame,
	future_frame: pd.DataFrame,
	overlay_prediction: pd.DataFrame,
	forecast_origin: pd.Timestamp,
	title: str,
	annotate_forecast_values: bool,
) -> None:
	ax.plot(history_frame["week_start"], history_frame["sales_units"], label="history",
	        color="#1f4e79", linewidth=2.2)
	ax.plot(future_frame["week_start"], future_frame["sales_units"], label="actual future",
	        color="#2a9d8f", linewidth=2.2, marker="o", markersize=4)
	ax.plot(
		overlay_prediction["target_date"],
		overlay_prediction["prediction"],
		label=f"forecast ({overlay_prediction['model_name'].iloc[0]})",
		color="#d1495b",
		linewidth=2.2,
		marker="o",
	)
	ax.fill_between(
		overlay_prediction["target_date"],
		overlay_prediction["actual"],
		overlay_prediction["prediction"],
		color="#f4a261",
		alpha=0.18,
		label="forecast gap",
	)
	ax.axvline(forecast_origin, color="#5c677d", linestyle="--", linewidth=1.4,
	           label="forecast origin")
	set_axis_labels(ax, title=title, xlabel="week_start", ylabel="sales_units")
	ax.legend(
		[
			"История",
			"Фактическое будущее",
			f"Прогноз",
			"Зазор прогноза",
			"Дата прогноза",
		]
	)
	if annotate_forecast_values:
		for row in overlay_prediction.itertuples(index=False):
			ax.annotate(
				f"p={row.prediction:.1f}\na={row.actual:.1f}",
				(row.target_date, row.prediction),
				textcoords="offset points",
				xytext=(0, 8),
				ha="center",
				fontsize=8,
				color="#7a1f2b",
			)
	local_values = pd.concat(
		[
			history_frame["sales_units"].astype(float),
			future_frame["sales_units"].astype(float),
			overlay_prediction["prediction"].astype(float),
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


def _plot_forecast_window_axis(
	ax: plt.Axes,
	history_frame: pd.DataFrame,
	future_frame: pd.DataFrame,
	overlay_prediction: pd.DataFrame,
	forecast_origin: pd.Timestamp,
	annotate_forecast_values: bool,
) -> None:
	ax.plot(
		history_frame["week_start"],
		history_frame["sales_units"],
		color="#8d99ae",
		linewidth=2.4,
		marker="o",
		markersize=4,
		label="recent history",
	)
	ax.plot(
		future_frame["week_start"],
		future_frame["sales_units"],
		color="#2a9d8f",
		linewidth=3.0,
		marker="o",
		markersize=6,
		label="actual future",
	)
	ax.plot(
		overlay_prediction["target_date"],
		overlay_prediction["prediction"],
		color="#d1495b",
		linewidth=3.0,
		marker="o",
		markersize=6,
		label="forecast",
	)
	for row in overlay_prediction.itertuples(index=False):
		ax.vlines(
			row.target_date,
			min(row.actual, row.prediction),
			max(row.actual, row.prediction),
			color="#f4a261",
			linewidth=3.0,
			alpha=0.65,
		)
		if annotate_forecast_values:
			ax.annotate(
				f"{row.prediction:.0f}",
				(row.target_date, row.prediction),
				textcoords="offset points",
				xytext=(0, 10),
				ha="center",
				fontsize=9,
				color="#7a1f2b",
				fontweight="bold",
			)
			ax.annotate(
				f"{row.actual:.0f}",
				(row.target_date, row.actual),
				textcoords="offset points",
				xytext=(0, -16),
				ha="center",
				fontsize=9,
				color="#0d6e57",
				fontweight="bold",
			)

	ax.axvline(forecast_origin, color="#5c677d", linestyle="--", linewidth=1.5)
	set_axis_labels(
		ax,
		title="Окно прогноза: фактические значения и точки прогноза по неделям",
		xlabel="week_start",
		ylabel="sales_units",
	)
	ax.legend(["Недавняя история", "Факт", "Прогноз", "Абсолютная ошибка"], loc="upper left")
	ax.grid(alpha=0.25, linestyle="--")
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
	for label in ax.get_xticklabels():
		label.set_rotation(25)
		label.set_ha("right")


def _write_actual_vs_prediction_plot(predictions: pd.DataFrame, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(9.5, 7.5))
	scenarios = sorted(predictions["scenario_name"].unique().tolist())
	palette = ["#1f77b4", "#2a9d8f", "#d1495b", "#f4a261", "#6c757d"]
	for index, scenario_name in enumerate(scenarios):
		scenario_frame = predictions.loc[predictions["scenario_name"] == scenario_name]
		ax.scatter(
			scenario_frame["actual"],
			scenario_frame["prediction"],
			alpha=0.55,
			s=38,
			color=palette[index % len(palette)],
			label=scenario_name,
		)

	all_values = pd.concat(
		[predictions["actual"].astype(float), predictions["prediction"].astype(float)],
		ignore_index=True,
	)
	limit = float(all_values.max()) if not all_values.empty else 1.0
	ax.plot([0, limit], [0, limit], linestyle="--", color="#5c677d", linewidth=1.5)
	set_axis_labels(
		ax,
		title="Факт против прогноза по всем forecast points",
		xlabel="actual",
		ylabel="prediction",
	)
	ax.grid(alpha=0.2, linestyle="--")
	ax.legend(title="scenario", loc="upper left")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def _write_abs_error_by_horizon_plot(predictions: pd.DataFrame, output_path: Path) -> None:
	frame = predictions.copy()
	frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()
	pivot = (
		frame.groupby(["scenario_name", "horizon"], as_index=False)["abs_error"]
		.mean()
		.sort_values(["scenario_name", "horizon"])
	)

	fig, ax = plt.subplots(figsize=(10.5, 6.8))
	for scenario_name, scenario_frame in pivot.groupby("scenario_name", sort=True):
		ax.plot(
			scenario_frame["horizon"],
			scenario_frame["abs_error"],
			linewidth=2.6,
			marker="o",
			markersize=6,
			label=scenario_name,
		)
	set_axis_labels(
		ax,
		title="Средняя абсолютная ошибка по горизонтам",
		xlabel="horizon",
		ylabel="mean absolute error",
	)
	ax.set_xticks(sorted(pivot["horizon"].unique().tolist()))
	ax.grid(alpha=0.25, linestyle="--")
	ax.legend(title="scenario", loc="upper left")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def _write_scenario_wape_plot(metrics: pd.DataFrame, output_path: Path) -> None:
	if metrics.empty:
		return

	fig, ax = plt.subplots(figsize=(10.5, 6.8))
	for scenario_name, scenario_frame in metrics.groupby("scenario_name", sort=True):
		ax.plot(
			scenario_frame["horizon"],
			scenario_frame["WAPE"],
			linewidth=2.8,
			marker="o",
			markersize=6,
			label=scenario_name,
		)
	set_axis_labels(
		ax,
		title="WAPE по горизонтам и сценариям",
		xlabel="horizon",
		ylabel="WAPE",
	)
	ax.set_xticks(sorted(metrics["horizon"].unique().tolist()))
	ax.grid(alpha=0.25, linestyle="--")
	ax.legend(title="scenario", loc="upper left")
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def _resolve_overlay_predictions(predictions: pd.DataFrame,
                                 overlay_series_id: str | None) -> pd.DataFrame:
	if predictions.empty:
		raise ValueError()

	if overlay_series_id is None:
		scenario_name = str(predictions["scenario_name"].iloc[0])
		series_id = str(predictions["series_id"].iloc[0])
		return predictions.loc[
			(predictions["scenario_name"] == scenario_name) & (predictions["series_id"] == series_id)
			].copy()

	matched = predictions.loc[predictions["series_id"] == overlay_series_id].copy()
	if matched.empty:
		scenario_name = str(predictions["scenario_name"].iloc[0])
		series_id = str(predictions["series_id"].iloc[0])
		return predictions.loc[
			(predictions["scenario_name"] == scenario_name) & (predictions["series_id"] == series_id)
			].copy()
	first_scenario = str(matched["scenario_name"].iloc[0])
	return matched.loc[matched["scenario_name"] == first_scenario].copy()


def build_report(
	model_name: str,
	frame: pd.DataFrame,
	predictions: pd.DataFrame,
	metrics: pd.DataFrame,
	artifact_path: str | Path | None,
) -> str:
	scenarios = sorted(predictions["scenario_name"].unique().tolist())
	lines = [
		"# Synthetic Prediction Report",
		"",
		f"- Модель: {model_name}",
		f"- Сценарии: {', '.join(scenarios)}",
		f"- Рядов с прогнозом: {predictions['series_id'].nunique()}",
		f"- Forecast rows: {len(predictions)}",
	]
	if artifact_path is not None:
		lines.append(f"- Источник модели: saved artifact `{artifact_path}`")
	if "scenario_name" in frame.columns:
		lines.extend(["", "## Scenario coverage"])
		for scenario_name, scenario_frame in frame.groupby("scenario_name", sort=True):
			lines.append(
				f"- {scenario_name}: {scenario_frame['series_id'].nunique()} series, {len(scenario_frame)} rows"
			)
	if not metrics.empty:
		row = metrics.sort_values(["WAPE", "scenario_name", "horizon"]).iloc[0]
		lines.extend(
			[
				"",
				"## Best diagnostic row",
				f"- scenario_name: {row['scenario_name']}",
				f"- horizon: {int(row['horizon'])}",
				f"- WAPE: {row['WAPE']:.4f}",
			]
		)
	lines.append("")
	return "\n".join(lines)


def prepare_predict_frame(
	dataset_path: str | Path,
	scenario_name: str | None,
	aggregation_level: str | None = None,
) -> pd.DataFrame:
	frame = pd.read_csv(dataset_path, parse_dates=["week_start"])
	required_columns = {"series_id", "category", "week_start", "is_history", "sales_units"}
	missing = required_columns.difference(frame.columns)
	if missing:
		raise ValueError()
	if scenario_name is not None:
		if "scenario_name" not in frame.columns:
			raise ValueError()
		frame = frame.loc[frame["scenario_name"] == scenario_name].copy()
		if frame.empty:
			raise ValueError()
	if aggregation_level is not None:
		frame = _aggregate_predict_frame(frame, aggregation_level)
	sort_columns = ["series_id", "week_start"]
	if "scenario_name" in frame.columns:
		sort_columns = ["scenario_name", "series_id", "week_start"]
	return frame.sort_values(sort_columns).reset_index(drop=True)


def _aggregate_predict_frame(frame: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
	if aggregation_level == "sku":
		return frame.copy()
	if aggregation_level != "category":
		raise ValueError()

	group_columns = ["category", "week_start", "is_history"]
	if "scenario_name" in frame.columns:
		group_columns = ["scenario_name", *group_columns]

	aggregations: dict[str, str] = {"sales_units": "sum"}
	if "stockout_flag" in frame.columns:
		aggregations["stockout_flag"] = "max"
	if "demand_noise_scale" in frame.columns:
		aggregations["demand_noise_scale"] = "mean"
	if "target_name" in frame.columns:
		aggregations["target_name"] = "first"

	aggregated = frame.groupby(group_columns, as_index=False).agg(aggregations)
	aggregated["series_id"] = aggregated["category"]
	return aggregated


def resolve_scenario_value(group_key: object, series_frame: pd.DataFrame) -> str:
	if isinstance(group_key, tuple):
		return str(group_key[0])

	if "scenario_name" in series_frame.columns:
		return str(series_frame["scenario_name"].iloc[0])

	return "base"


def resolve_series_id(group_key: object) -> str:
	if isinstance(group_key, tuple):
		return str(group_key[-1])

	return str(group_key)
