from pathlib import Path

import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.experiment.experiment_artifact import ExperimentModelArtifactPayload
from mt.domain.model.model_result import ModelResult
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.model.model_config_manifest import serialize_model_config
from mt.domain.probabilistic.probabilistic import (
	ProbabilisticColumn,
	QUANTILE_COLUMNS,
)
from mt.infra.artifact.text_writer import write_csv, write_markdown, write_yaml
from mt.infra.experiment_artifact.models.storage_policy import (
	build_experiment_model_registry_aliases,
	build_experiment_model_registry_description,
	build_experiment_model_registry_tags,
	resolve_experiment_model_registry_name,
)
from mt.infra.model_artifact.model.fitting import save_trained_model_bundle
from mt.infra.tracking.backend_builder import build_tracking_backend
from mt.infra.tracking.backend_resolver import resolve_tracking_backend_name


def write_experiment_model_artifacts(
	ctx: ExperimentPipelineContext,
) -> None:
	selected_payload = _require_selected_payload(ctx)
	_write_single_experiment_model_artifacts(
		ctx=ctx,
		model_dir=ctx.artifacts_paths_map.model,
		payload=selected_payload,
	)


def _write_single_experiment_model_artifacts(
	ctx: ExperimentPipelineContext,
	model_dir: Path,
	payload: ExperimentModelArtifactPayload,
) -> None:
	result = payload.result
	best_model_bundle = ctx.require_best_model_bundle()
	model_dir.mkdir(parents=True, exist_ok=True)

	_write_prediction_artifacts(
		model_dir,
		result,
		payload.metrics_overall,
		payload.metrics_by_horizon,
		payload.probabilistic_metrics_overall,
		payload.probabilistic_metrics_by_horizon,
	)

	manifest_payload = _build_model_manifest_payload(ctx, result, payload.model_manifest)

	write_yaml(model_dir / "manifest.yaml", manifest_payload)
	write_markdown(
		model_dir / "summary.md",
		_build_model_run_summary(
			result,
			payload.metrics_overall,
			payload.probabilistic_metrics_overall,
		),
	)
	selected_metrics = _build_registry_metrics_payload(
		result.info.model_name,
		payload.metrics_overall,
		payload.probabilistic_metrics_overall,
	)
	registry_name = resolve_experiment_model_registry_name(ctx)
	registry_tags = build_experiment_model_registry_tags(ctx, result, selected_metrics)
	registry_aliases = build_experiment_model_registry_aliases()
	registry_description = build_experiment_model_registry_description(
		ctx,
		result,
		registry_name,
		selected_metrics,
	)
	save_trained_model_bundle(
		output_dir=model_dir,
		trained_bundle=best_model_bundle,
	)

	tracking_backend = build_tracking_backend(
		resolve_tracking_backend_name(
			ctx.observability.execution_mode if ctx.observability else None
		)
	)
	tracking_backend.log_experiment_model(
		parent_run_id=str(ctx.runtime_metadata.get("mlflow_parent_run_id") or "") or None,
		model_name=result.info.model_name,
		model_dir=model_dir,
		selected_metrics=selected_metrics,
		registry_name=registry_name,
		registry_tags=registry_tags,
		registry_aliases=registry_aliases,
		registry_description=registry_description,
	)

	tracking_backend.log_experiment_model_run(
		parent_run_id=str(ctx.runtime_metadata.get("mlflow_parent_run_id") or "") or None,
		model_name=result.info.model_name,
		model_family=result.info.model_family,
		model_dir=model_dir,
		model_manifest=manifest_payload,
		metrics_overall=payload.metrics_overall,
		metrics_by_horizon=payload.metrics_by_horizon,
		probabilistic_metrics_overall=payload.probabilistic_metrics_overall,
		probabilistic_metrics_by_horizon=payload.probabilistic_metrics_by_horizon,
		used_feature_columns=result.used_feature_columns,
		calibration_summary=result.calibration_summary,
		probabilistic_metadata=result.probabilistic_metadata,
		train_time_seconds=result.train_time_seconds,
		inference_time_seconds=result.inference_time_seconds,
		wall_time_seconds=result.wall_time_seconds,
	)


def _resolve_selected_payload(
	ctx: ExperimentPipelineContext,
) -> ExperimentModelArtifactPayload | None:
	for payload in ctx.model_artifact_payloads:
		if str(payload.result.info.model_name) == str(ctx.selected_model_name):
			return payload
	return None


def _require_selected_payload(
	ctx: ExperimentPipelineContext,
) -> ExperimentModelArtifactPayload:
	selected_model_name = ctx.require_selected_model_name()
	selected_payload = _resolve_selected_payload(ctx)
	if selected_payload is None:
		raise ValueError(
			f"Selected experiment model payload is not available for {selected_model_name}"
		)
	return selected_payload


def _write_prediction_artifacts(
	model_dir: Path,
	result: ModelResult,
	metrics_overall: pd.DataFrame,
	metrics_by_horizon: pd.DataFrame,
	probabilistic_metrics_overall: pd.DataFrame,
	probabilistic_metrics_by_horizon: pd.DataFrame,
) -> None:
	predictions = result.predictions
	base_columns = [
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
	quantile_columns = [
		*base_columns,
		*QUANTILE_COLUMNS,
		ProbabilisticColumn.SOURCE,
		ProbabilisticColumn.STATUS,
	]
	interval_columns = [
		*base_columns,
		ProbabilisticColumn.LO_80,
		ProbabilisticColumn.HI_80,
		ProbabilisticColumn.LO_95,
		ProbabilisticColumn.HI_95,
		ProbabilisticColumn.SOURCE,
		ProbabilisticColumn.STATUS,
	]
	write_csv(model_dir / "raw_predictions.csv", predictions)
	write_csv(model_dir / "quantile_predictions.csv", predictions.loc[:, quantile_columns])
	write_csv(model_dir / "interval_predictions.csv", predictions.loc[:, interval_columns])
	write_csv(model_dir / "metrics_overall.csv", metrics_overall)
	write_csv(model_dir / "metrics_by_horizon.csv", metrics_by_horizon)
	write_csv(model_dir / "probabilistic_metrics_overall.csv", probabilistic_metrics_overall)
	write_csv(model_dir / "probabilistic_metrics_by_horizon.csv", probabilistic_metrics_by_horizon)
	if result.calibration_summary is not None:
		write_csv(model_dir / "calibration_summary.csv", result.calibration_summary)


def _build_model_manifest_payload(
	ctx: ExperimentPipelineContext,
	result: ModelResult,
	model_manifest: ModelManifest,
) -> dict[str, object]:
	dataset = ctx.require_dataset()
	return {
		"aggregation_level": dataset.aggregation_level,
		"model_name": result.info.model_name,
		"model_family": result.info.model_family,
		"features": _serialize_feature_manifest(model_manifest),
		"config": serialize_model_config(model_manifest.config),
		"used_feature_columns": result.used_feature_columns,
		"probabilistic_metadata": result.probabilistic_metadata,
		"seed": ctx.manifest.runtime.seed,
	}


def _serialize_feature_manifest(model_manifest: ModelManifest) -> dict[str, object]:
	features = model_manifest.features
	if not features.enabled:
		return {"enabled": False}

	payload: dict[str, object] = {"enabled": True}
	defaults = ModelManifest(name=model_manifest.name, config=model_manifest.config).features

	for field_name in (
		"feature_set",
		"lags",
		"rolling_windows",
		"use_calendar",
		"use_category_encodings",
	):
		value = getattr(features, field_name)
		default_value = getattr(defaults, field_name)
		if value != default_value:
			payload[field_name] = value

	return payload


def _build_model_run_summary(
	result: ModelResult,
	metrics_overall: pd.DataFrame,
	probabilistic_metrics_overall: pd.DataFrame,
) -> list[str]:
	row = metrics_overall.iloc[0]
	prob_row = (
		probabilistic_metrics_overall.iloc[0]
		if not probabilistic_metrics_overall.empty
		else None
	)
	wis_value = (
		f"{float(prob_row['WIS']):.4f}"
		if prob_row is not None and pd.notna(prob_row["WIS"])
		else "n/a"
	)
	coverage80_value = (
		f"{float(prob_row['Coverage80']):.4f}"
		if prob_row is not None and pd.notna(prob_row["Coverage80"])
		else "n/a"
	)
	realized_source = _resolve_realized_probabilistic_source(result.predictions)
	return [
		f"# {result.info.model_name}",
		"",
		f"- Число использованных признаков: {len(result.used_feature_columns)}",
		f"- Время обучения, с: {_format_seconds(result.train_time_seconds)}",
		f"- Время инференса, с: {_format_seconds(result.inference_time_seconds)}",
		f"- Полное время модели, с: {_format_seconds(result.wall_time_seconds)}",
		"",
		"## Метрики",
		f"- WAPE: {row['WAPE']:.4f}",
		f"- sMAPE: {row['sMAPE']:.4f}",
		f"- MAE: {row['MAE']:.4f}",
		f"- Bias: {row['Bias']:.4f}",
		"",
		"## Probabilistic",
		f"- Source: {realized_source}",
		f"- Lower bounds clipped to zero: {bool(result.probabilistic_metadata.get('lower_bounds_clipped_to_zero', False))}",
		f"- Quantile crossing corrected: {bool(result.probabilistic_metadata.get('quantile_crossing_corrected', False))}",
		f"- WIS: {wis_value}",
		f"- Coverage80: {coverage80_value}",
	]


def _format_seconds(value: float | None) -> str:
	if value is None:
		return "n/a"
	return f"{value:.3f}"


def _resolve_realized_probabilistic_source(predictions: pd.DataFrame) -> str:
	if predictions.empty or "probabilistic_status" not in predictions.columns:
		return "none"
	available = predictions[predictions["probabilistic_status"].astype(str) == "available"]
	if available.empty:
		return "none"
	sources = sorted(set(available["probabilistic_source"].astype(str).tolist()))
	return ",".join(sources)


def _build_registry_metrics_payload(
	model_name: str,
	metrics_overall: pd.DataFrame,
	probabilistic_metrics_overall: pd.DataFrame,
) -> dict[str, object]:
	payload: dict[str, object] = {"model_name": model_name}
	if not metrics_overall.empty:
		for key, value in metrics_overall.iloc[0].to_dict().items():
			if key == "model_name":
				continue
			payload[key] = value
	if not probabilistic_metrics_overall.empty:
		for key, value in probabilistic_metrics_overall.iloc[0].to_dict().items():
			if key == "model_name":
				continue
			payload[key] = value
	return payload
