from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.domain.metric.metric_name import POINT_METRIC_NAMES
from mt.domain.model.model_result import ModelResult
from mt.infra.observability.logger.runtime_logger import log_debug, log_info
from mt.infra.observability.runtime.event_emitter import emit_event


def log_stage_start(stage_name: str) -> None:
	log_info("start", scope="stage", name=stage_name)
	emit_event("stage_started", {"stage_name": stage_name})


def log_stage_end(stage_name: str, wall_time_seconds: float) -> None:
	log_info("done", scope="stage", name=stage_name, wall_time=f"{wall_time_seconds:.3f}s")
	emit_event("stage_completed", {"stage_name": stage_name, "wall_time_seconds": wall_time_seconds})


def log_experiment_start(manifest: ExperimentPipelineManifest) -> None:
	feature_manifest = manifest.build_combined_feature_manifest()
	log_info(
		"start",
		scope="experiment",
		aggregation=manifest.dataset.aggregation_level,
		feature_set=feature_manifest.feature_set,
		models=",".join(manifest.enabled_model_names),
		artifacts_dir=str(manifest.runtime.artifacts_dir),
	)


def log_experiment_end(ctx: Any) -> None:
	wall_time = (
		f"{ctx.pipeline_wall_time_seconds:.3f}s"
		if ctx.pipeline_wall_time_seconds is not None
		else "n/a"
	)
	artifacts_dir = str(ctx.runtime_metadata.get("logical_artifact_root") or ctx.artifacts_paths_map.root)
	log_info("done", scope="experiment", wall_time=wall_time, artifacts_dir=artifacts_dir)
	emit_event(
		"experiment_completed",
		{
			"wall_time_seconds": ctx.pipeline_wall_time_seconds,
			"artifacts_dir": artifacts_dir,
		},
	)


def log_experiment_metrics(ctx: Any) -> None:
	if ctx.overall_metrics.empty:
		log_info("unavailable", scope="metrics")
		return

	display_columns = [
		column
		for column in ("model_name", *POINT_METRIC_NAMES)
		if column in ctx.overall_metrics.columns
	]
	metrics_table = ctx.overall_metrics.loc[:, display_columns].copy()
	for column in [column for column in display_columns if column != "model_name"]:
		metrics_table[column] = metrics_table[column].map(lambda value: f"{value:.6f}")
	log_info("final", scope="metrics", table=metrics_table.to_string(index=False))


def log_model_runner_start(model_name: str, windows: pd.DataFrame) -> None:
	log_info("start", scope="model", name=model_name, windows=len(windows))
	emit_event("model_started", {"model_name": model_name, "windows": len(windows)})


def log_model_runner_end(result: ModelResult, model_dir: Path) -> None:
	log_info(
		"done",
		scope="model",
		name=result.info.model_name,
		family=result.info.model_family,
		train_time=f"{(result.train_time_seconds or 0.0):.3f}s",
		infer_time=f"{(result.inference_time_seconds or 0.0):.3f}s",
		wall_time=f"{(result.wall_time_seconds or 0.0):.3f}s",
		artifacts=str(model_dir),
	)
	emit_event(
		"model_completed",
		{
			"model_name": result.info.model_name,
			"model_family": result.info.model_family,
			"train_time_seconds": result.train_time_seconds,
			"inference_time_seconds": result.inference_time_seconds,
			"wall_time_seconds": result.wall_time_seconds,
			"artifacts_dir": str(model_dir),
		},
	)


def log_model_fit_and_predict_horizon_start(
	model_name: str,
	horizon: int,
	horizon_windows: pd.DataFrame,
) -> None:
	log_horizon_start(model_name=model_name, horizon=horizon, windows=len(horizon_windows))


def log_horizon_start(model_name: str, horizon: int, windows: int) -> None:
	log_info("start", scope="horizon", model=model_name, h=horizon, windows=windows)


def log_model_fit_and_predict_horizon_end(
	model_name: str,
	horizon: int,
	forecast_origin: pd.Timestamp,
	prediction_rows: int,
) -> None:
	log_debug(
		"predicted",
		scope="horizon",
		model=model_name,
		h=horizon,
		origin=str(forecast_origin),
		rows=prediction_rows,
	)


def log_model_save(model_name: str, horizons: list[str], root: Path) -> None:
	log_info(
		"saved",
		scope="artifact",
		model=model_name,
		horizons=",".join(str(horizon) for horizon in horizons),
		path=str(root),
	)
	emit_event(
		"model_saved",
		{"model_name": model_name, "horizons": [str(h) for h in horizons], "path": str(root)},
	)


def log_model_prediction(model_name: str, result: Any, output_path: Path) -> None:
	rows = len(result) if hasattr(result, "__len__") else result
	log_info("saved", scope="forecast", model=model_name, rows=rows, path=str(output_path))
	emit_event("forecast_saved", {"model_name": model_name, "rows": rows, "path": str(output_path)})


def log_bootstrap_ci(model_a: Any, model_b: Any, blocks: Any, n_bootstrap: Any) -> None:
	log_info("ci", scope="bootstrap", model_a=model_a, model_b=model_b, blocks=len(blocks), n_bootstrap=n_bootstrap)


def log_bootstrap_ci_fast(model_a: Any, model_b: Any, blocks: Any, n_bootstrap: Any) -> None:
	log_info(
		"ci_fast",
		scope="bootstrap",
		model_a=model_a,
		model_b=model_b,
		blocks=len(blocks),
		n_bootstrap=n_bootstrap,
	)
