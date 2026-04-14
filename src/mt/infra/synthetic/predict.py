from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import yaml

from mt.domain.model import BestModelArtifact
from mt.domain.manifest import DLManifest, FeatureManifest
from mt.infra.artifact.plot_labels import set_axis_labels
from mt.infra.feature.segmentation import segment_series
from mt.infra.feature.supervised_builder import make_supervised_frame
from mt.infra.model.best_model_artifact import load_best_model_artifact
from mt.infra.model.runner import run_model
from mt.infra.metric.calculates import calculate_metrics

ALLOWED_PREDICT_MODELS = {"naive", "seasonal_naive", "moving_average", "best_auto"}


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


@dataclass(slots=True)
class ReferenceModelConfig:
	"""Ссылка на сохраненную лучшую модель из experiment artifacts"""

	model_name: str
	artifact: BestModelArtifact | None
	feature_manifest: FeatureManifest
	model_params: dict[str, object]
	dl_manifest: DLManifest | None
	source_dir: Path


def build_saved_model_predictions(
	frame: pd.DataFrame,
	reference_model: ReferenceModelConfig,
	horizon_weeks: int,
) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	scenario_groups: list[tuple[str, pd.DataFrame]]

	if "scenario_name" in frame.columns:
		scenario_groups = [
			(str(scenario_name),
			 scenario_frame.sort_values(["series_id", "week_start"]).reset_index(drop=True))
			for scenario_name, scenario_frame in frame.groupby("scenario_name", sort=True)
		]
	else:
		scenario_groups = [
			("base", frame.sort_values(["series_id", "week_start"]).reset_index(drop=True))]

	for scenario_name, scenario_frame in scenario_groups:
		try:
			forecast_rows = (
				run_saved_model_forecast_window(
					scenario_frame=scenario_frame,
					horizon_weeks=horizon_weeks,
					artifact=reference_model.artifact,
				)
				if reference_model.artifact is not None
				else pd.DataFrame()
			)
		except Exception:
			forecast_rows = pd.DataFrame()

		if _should_fallback_to_artifact_refit(forecast_rows, horizon_weeks):
			forecast_rows = run_direct_forecast_window(
				scenario_frame=scenario_frame,
				horizon_weeks=horizon_weeks,
				feature_manifest=reference_model.feature_manifest,
				model_name=reference_model.model_name,
				model_params=reference_model.model_params,
				dl_manifest=reference_model.dl_manifest,
			)

		if forecast_rows.empty:
			continue

		forecast_rows["scenario_name"] = scenario_name
		rows.append(forecast_rows)

	if not rows:
		return pd.DataFrame()

	return pd.concat(rows, ignore_index=True).sort_values(
		["scenario_name", "horizon", "series_id"]).reset_index(drop=True)


def run_saved_model_forecast_window(
	scenario_frame: pd.DataFrame,
	horizon_weeks: int,
	artifact: BestModelArtifact,
) -> pd.DataFrame:
	history_weekly = prepare_history_weekly(scenario_frame)
	full_weekly = prepare_full_weekly(scenario_frame)
	segments = segment_series(history_weekly)
	supervised, _ = make_supervised_frame(full_weekly, segments, artifact.feature_manifest)

	for horizon in artifact.horizons:
		target_column = f"target_h{horizon}"
		if target_column not in supervised.columns:
			supervised[target_column] = supervised.groupby("series_id")["sales_units"].shift(-horizon)

	history_weeks = sorted(pd.to_datetime(
		scenario_frame.loc[scenario_frame["is_history"].astype(bool), "week_start"].unique()
	))
	if not history_weeks:
		raise ValueError()
	forecast_origin = pd.Timestamp(history_weeks[-1])

	rows: list[dict[str, object]] = []
	for horizon in sorted(h for h in artifact.horizons if h <= horizon_weeks):
		adapter = artifact.adapters_by_horizon.get(horizon)
		if adapter is None:
			continue

		target_column = f"target_h{horizon}"
		prepared_frame = adapter.prepare_frame(supervised)
		predict_candidates = prepared_frame.loc[prepared_frame["week_start"] == forecast_origin].copy()
		predict_frame = adapter.select_inference_frame(predict_candidates, artifact.feature_columns)
		predict_frame = predict_frame.dropna(subset=[target_column]).copy()
		if predict_frame.empty:
			continue

		predictions = adapter.predict(
			predict_frame=predict_frame,
			feature_columns=artifact.feature_columns,
			target_column=target_column,
			horizon=horizon,
		)
		if len(predictions) != len(predict_frame):
			raise ValueError()

		target_date = forecast_origin + pd.Timedelta(weeks=horizon)
		for (_, row), prediction in zip(predict_frame.iterrows(), predictions, strict=False):
			rows.append(
				{
					"model_name": artifact.model_name,
					"model_family": adapter.get_model_info().model_family,
					"series_id": row["series_id"],
					"category": row["category"],
					"segment_label": row.get("segment_label"),
					"forecast_origin": forecast_origin,
					"target_date": target_date,
					"horizon": horizon,
					"actual": float(row[target_column]),
					"prediction": max(float(prediction), 0.0),
				}
			)

	if not rows:
		return pd.DataFrame(
			columns=[
				"model_name",
				"model_family",
				"series_id",
				"category",
				"segment_label",
				"forecast_origin",
				"target_date",
				"horizon",
				"actual",
				"prediction",
			]
		)

	return pd.DataFrame(rows).sort_values(["horizon", "series_id"]).reset_index(drop=True)


def _should_fallback_to_artifact_refit(predictions: pd.DataFrame, horizon_weeks: int) -> bool:
	if predictions.empty or predictions["prediction"].isna().any():
		return True

	observed_horizons = {int(value) for value in predictions["horizon"].unique().tolist()}
	expected_horizons = set(range(1, horizon_weeks + 1))

	return not expected_horizons.issubset(observed_horizons)


def run_direct_forecast_window(
	scenario_frame: pd.DataFrame,
	horizon_weeks: int,
	feature_manifest: FeatureManifest,
	model_name: str,
	model_params: dict[str, object] | None = None,
	dl_manifest: DLManifest | None = None,
) -> pd.DataFrame:
	full_weekly = prepare_full_weekly(scenario_frame)
	supervised, feature_columns = build_supervised_with_targets(
		full_weekly,
		feature_manifest,
		horizon_weeks
	)
	forecast_windows = build_future_windows(scenario_frame, horizon_weeks)

	result = run_model(
		model_name=model_name,
		supervised=supervised,
		feature_columns=feature_columns,
		windows=forecast_windows,
		seed=42,
		model_params=model_params,
		dl_manifest=dl_manifest,
	)

	if result.predictions.empty:
		return pd.DataFrame(
			columns=["model_name", "series_id", "category", "forecast_origin", "target_date", "horizon",
			         "actual", "prediction"]
		)

	return result.predictions


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for (scenario_name, horizon), horizon_frame in predictions.groupby(["scenario_name", "horizon"],
	                                                                   sort=True):
		rows.append(
			{
				"scenario_name": scenario_name,
				"horizon": int(horizon),
				**calculate_metrics(horizon_frame),
			}
		)
	return pd.DataFrame(rows)


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
			f"Прогноз ({overlay_prediction['model_name'].iloc[0]})",
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
		"",
		"## Ограничения",
		"- В качестве первичного источника используется saved `model.pkl`; при невозможности прямого synthetic inference pipeline откатывается к artifact-guided refit того же типа модели.",
		"- Текущая команда ожидает колонку `is_history`, чтобы честно разделить history и future actual.",
		"- Overlay отражает один выбранный ряд; для остальных рядов сохраняется только CSV-прогноз и метрики.",
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
		best_row = metrics.sort_values(["WAPE", "scenario_name", "horizon"]).iloc[0]
		lines.extend(
			[
				"",
				"## Best diagnostic row",
				f"- scenario_name: {best_row['scenario_name']}",
				f"- horizon: {int(best_row['horizon'])}",
				f"- WAPE: {best_row['WAPE']:.4f}",
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
	if "price" in frame.columns:
		aggregations["price"] = "mean"
	if "promo_planned" in frame.columns:
		aggregations["promo_planned"] = "mean"
	if "promo_actual" in frame.columns:
		aggregations["promo_actual"] = "mean"
	if "discount_rate" in frame.columns:
		aggregations["discount_rate"] = "mean"
	if "base_price" in frame.columns:
		aggregations["base_price"] = "mean"
	if "stockout_flag" in frame.columns:
		aggregations["stockout_flag"] = "max"
	if "demand_noise_scale" in frame.columns:
		aggregations["demand_noise_scale"] = "mean"
	if "promo_known_in_advance" in frame.columns:
		aggregations["promo_known_in_advance"] = "all"
	if "price_known_in_advance" in frame.columns:
		aggregations["price_known_in_advance"] = "all"
	if "promo_covariate_class" in frame.columns:
		aggregations["promo_covariate_class"] = "first"
	if "price_covariate_class" in frame.columns:
		aggregations["price_covariate_class"] = "first"
	if "target_name" in frame.columns:
		aggregations["target_name"] = "first"

	aggregated = frame.groupby(group_columns, as_index=False).agg(aggregations)
	aggregated["series_id"] = aggregated["category"]
	return aggregated


def infer_horizon(frame: pd.DataFrame) -> int:
	horizon = int((~frame["is_history"].astype(bool)).sum() / frame["series_id"].nunique())
	if horizon < 1:
		raise ValueError()
	return horizon


def load_reference_model_config(artifact_path: str | Path) -> ReferenceModelConfig:
	artifact_path = Path(artifact_path)
	best_model_dir = artifact_path if artifact_path.is_dir() else artifact_path.parent
	artifact_manifest_path = best_model_dir / "artifact_manifest.yaml"
	if not artifact_manifest_path.exists():
		raise FileNotFoundError(f)

	artifact_manifest = yaml.safe_load(artifact_manifest_path.read_text(encoding="utf-8")) or {}
	model_name = str(artifact_manifest.get("model_name", "")).strip()
	model_pkl_path = best_model_dir / "model.pkl"
	model_run_manifest_path = best_model_dir.parent / model_name / "run_manifest.yaml"
	run_manifest = yaml.safe_load(model_run_manifest_path.read_text(encoding="utf-8")) or {}
	features_data = run_manifest.get("features", artifact_manifest.get("feature_manifest", {}))
	params_data = run_manifest.get("params", {})
	dl_data = run_manifest.get("dl")
	if not isinstance(dl_data, dict):
		config_data = run_manifest.get("config", {})
		if isinstance(config_data, dict):
			dl_candidate = {
				field_name: config_data[field_name]
				for field_name in DLManifest.__dataclass_fields__
				if field_name in config_data
			}
			dl_data = dl_candidate if dl_candidate else None

	return ReferenceModelConfig(
		model_name=model_name,
		artifact=load_best_model_artifact(model_pkl_path),
		feature_manifest=FeatureManifest(**features_data),
		model_params=dict(params_data) if isinstance(params_data, dict) else {},
		dl_manifest=DLManifest(**dl_data) if isinstance(dl_data, dict) else None,
		source_dir=best_model_dir,
	)


def prepare_history_weekly(series_frame: pd.DataFrame) -> pd.DataFrame:
	history = series_frame.loc[series_frame["is_history"].astype(bool)].copy()
	if history.empty:
		raise ValueError()

	history["promo"] = history["promo_planned"].astype(
		float) if "promo_planned" in history.columns else 0.0

	if "price" not in history.columns:
		history["price"] = np.nan

	return history.loc[:, ["series_id", "category", "week_start", "sales_units", "price", "promo"]]


def prepare_full_weekly(series_frame: pd.DataFrame) -> pd.DataFrame:
	full = series_frame.copy()
	full["promo"] = full["promo_planned"].astype(float) if "promo_planned" in full.columns else 0.0

	if "price" not in full.columns:
		full["price"] = np.nan

	return full.loc[:, ["series_id", "category", "week_start", "sales_units", "price", "promo"]]


def build_supervised_with_targets(
	weekly: pd.DataFrame,
	feature_manifest: FeatureManifest,
	horizon_weeks: int,
) -> tuple[pd.DataFrame, list[str]]:
	segments = segment_series(weekly)
	supervised, feature_columns = make_supervised_frame(weekly, segments, feature_manifest)

	for horizon in range(1, horizon_weeks + 1):
		supervised[f"target_h{horizon}"] = supervised.groupby("series_id")["sales_units"].shift(
			-horizon)

	return supervised, feature_columns


def build_validation_windows(history_weekly: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
	unique_weeks = sorted(pd.to_datetime(history_weekly["week_start"].unique()))
	last_history_week = pd.Timestamp(unique_weeks[-1])

	rows: list[dict[str, object]] = []
	for horizon in range(1, horizon_weeks + 1):
		forecast_origin = pd.Timestamp(unique_weeks[-1 - horizon])
		train_end = pd.Timestamp(unique_weeks[-1 - 2 * horizon])
		rows.append(
			{
				"horizon": horizon,
				"forecast_origin": forecast_origin,
				"train_end": train_end,
				"test_start": last_history_week,
			}
		)

	return pd.DataFrame(rows)


def build_future_windows(scenario_frame: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
	history_weeks = sorted(pd.to_datetime(
		scenario_frame.loc[scenario_frame["is_history"].astype(bool), "week_start"].unique()
	))
	forecast_origin = pd.Timestamp(history_weeks[-1])

	rows: list[dict[str, object]] = []
	for horizon in range(1, horizon_weeks + 1):
		rows.append(
			{
				"horizon": horizon,
				"forecast_origin": forecast_origin,
				"train_end": forecast_origin - pd.Timedelta(weeks=horizon),
				"test_start": forecast_origin + pd.Timedelta(weeks=horizon),
			}
		)

	return pd.DataFrame(rows)


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
