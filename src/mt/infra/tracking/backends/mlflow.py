from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from mt.domain.metric.metric_name import POINT_METRIC_NAMES, PROBABILISTIC_METRIC_NAMES
from mt.domain.tracking.tracking_backend import TrackingBackend
from mt.domain.tracking.tracking_request import TrackingRequest
from mt.domain.tracking.tracking_run import TrackingRunHandle
from mt.infra.artifact.text_writer import write_yaml
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.helper.model_registry import normalize_registry_aliases
from mt.infra.tracking.note_renderer import render_mlflow_note
from mt.infra.tracking.payload_adapter import stringify_param
from mt.infra.tracking.report_metrics import parse_report_file


@dataclass(slots=True, frozen=True)
class MlflowTrackingSettings:
	tracking_uri: str | None
	experiment_prefix: str

	@classmethod
	def from_env(cls) -> "MlflowTrackingSettings":
		return cls(
			tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
			experiment_prefix=os.getenv("MT_MLFLOW_EXPERIMENT_PREFIX", "mt"),
		)


class MlflowTrackingBackend(TrackingBackend):
	name = "mlflow"

	def start_run(self, request: TrackingRequest) -> TrackingRunHandle:
		return _start_pipeline_run(
			pipeline_type=request.pipeline_type,
			manifest_payload=request.manifest_payload,
			params=request.params,
			tags=request.tags,
		)

	def log_stage_run(
		self,
		run_id: str | None,
		stage_name: str,
		stage_timing: dict[str, object] | None,
		artifact_paths: list[Path],
		artifact_root: str | Path,
	) -> None:
		if run_id is None:
			return
		with mlflow.start_run(run_id=run_id):
			if stage_timing is not None:
				_log_dict_artifact(stage_timing, f"stage_runs/{stage_name}/timing.yaml")

			_log_files(
				artifact_root=artifact_root,
				artifact_paths=artifact_paths,
				artifact_prefix=f"stage_runs/{stage_name}/artifacts",
			)

	def log_pipeline_summary(
		self,
		run_id: str | None,
		metrics: dict[str, float],
		artifact_paths: list[Path],
		artifact_root: str | Path,
	) -> None:
		if run_id is None:
			return

		with mlflow.start_run(run_id=run_id):
			for metric_name, metric_value in metrics.items():
				mlflow.log_metric(metric_name, metric_value)

			report_path = Path(artifact_root) / "report" / "REPORT.md"

			_log_report_payload(report_path)
			_set_run_note_from_report(report_path)
			_log_profile_metrics_from_artifacts(Path(artifact_root))
			_log_files(
				artifact_root=artifact_root,
				artifact_paths=artifact_paths,
				artifact_prefix="artifacts",
			)

	def log_experiment_model_run(
		self,
		parent_run_id: str | None,
		model_name: str,
		model_family: str,
		model_dir: str | Path,
		model_manifest: dict[str, Any],
		metrics_overall: pd.DataFrame,
		metrics_by_horizon: pd.DataFrame,
		probabilistic_metrics_overall: pd.DataFrame,
		probabilistic_metrics_by_horizon: pd.DataFrame,
		used_feature_columns: list[str],
		calibration_summary: pd.DataFrame | None,
		probabilistic_metadata: dict[str, Any] | None,
		train_time_seconds: float | None,
		inference_time_seconds: float | None,
		wall_time_seconds: float | None,
	) -> None:
		if parent_run_id is None:
			return

		with mlflow.start_run(run_id=parent_run_id):
			with mlflow.start_run(
				run_name=f"model:{model_name}",
				nested=True,
				tags={
					"run_kind": "experiment_model",
					"model_name": model_name,
					"model_family": model_family,
				},
			):
				_log_params(
					{
						"model_name": model_name,
						"model_family": model_family,
						"feature_count": len(used_feature_columns),
						"train_time_seconds": train_time_seconds,
						"inference_time_seconds": inference_time_seconds,
						"wall_time_seconds": wall_time_seconds,
					},
				)
				_log_dict_artifact(
					{
						"model_manifest": model_manifest,
						"used_feature_columns": used_feature_columns,
						"probabilistic_metadata": probabilistic_metadata or {},
					},
					"metadata/model_details.yaml",
				)
				_log_frame_row_metrics("overall", metrics_overall, POINT_METRIC_NAMES)
				_log_frame_horizon_metrics("by_horizon", metrics_by_horizon, POINT_METRIC_NAMES)
				_log_frame_row_metrics(
					"probabilistic_overall",
					probabilistic_metrics_overall,
					PROBABILISTIC_METRIC_NAMES,
				)
				_log_frame_horizon_metrics(
					"probabilistic_by_horizon",
					probabilistic_metrics_by_horizon,
					PROBABILISTIC_METRIC_NAMES,
				)
				_log_calibration_metrics(calibration_summary)
				_log_probabilistic_metadata_metrics(probabilistic_metadata or {})
				_log_files(
					artifact_root=model_dir,
					artifact_paths=list(Path(model_dir).rglob("*")),
					artifact_prefix="artifacts",
				)

	def log_experiment_model(
		self,
		parent_run_id: str | None,
		model_name: str,
		model_dir: str | Path,
		selected_metrics: dict[str, Any],
		registry_name: str | None = None,
		registry_tags: dict[str, str] | None = None,
		registry_aliases: list[str] | None = None,
		registry_description: str | None = None,
	) -> None:
		if parent_run_id is None:
			return
		with mlflow.start_run(run_id=parent_run_id):
			mlflow.set_tag("model_name", model_name)
			for metric_name, metric_value in selected_metrics.items():
				if metric_name == "model_name":
					continue
				if isinstance(metric_value, (int, float)):
					mlflow.log_metric(f"model.{metric_name}", float(metric_value))
				artifact_path = "model"
			_log_model_bundle(model_dir=Path(model_dir), artifact_path=artifact_path)
			if registry_name:
				_try_register_model(
					run_id=parent_run_id,
					registry_name=registry_name,
					artifact_path=artifact_path,
					registry_tags=registry_tags or {},
					registry_aliases=registry_aliases or [],
					registry_description=registry_description,
				)


def log_model_storage_artifacts(
	run_id: str | None,
	model_dir: str | Path,
	artifact_path: str,
	registry_name: str | None = None,
	registry_tags: dict[str, str] | None = None,
	registry_aliases: list[str] | None = None,
	registry_description: str | None = None,
) -> None:
	if run_id is None:
		return

	with mlflow.start_run(run_id=run_id):
		_log_model_bundle(model_dir=Path(model_dir), artifact_path=artifact_path)
		if registry_name:
			_try_register_model(
				run_id=run_id,
				registry_name=registry_name,
				artifact_path=artifact_path,
				registry_tags=registry_tags or {},
				registry_aliases=registry_aliases or [],
				registry_description=registry_description,
			)


def _start_pipeline_run(
	pipeline_type: str,
	manifest_payload: dict[str, Any],
	params: dict[str, Any],
	tags: dict[str, str] | None = None,
) -> TrackingRunHandle:
	run_key = f"{pipeline_type}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
	started_at_utc = datetime.now(timezone.utc).isoformat()
	settings = MlflowTrackingSettings.from_env()
	experiment_name = _configure_mlflow(settings, pipeline_type)
	manifest_name = _resolve_manifest_name(pipeline_type, manifest_payload)

	with mlflow.start_run(
		run_name=f"{pipeline_type}:{manifest_name}:{run_key}",
		tags={
			"pipeline_type": pipeline_type,
			"manifest_name": manifest_name,
			"run_key": run_key,
			"started_at_utc": started_at_utc,
			**(tags or {}),
		},
	) as run:
		_log_params({"run_key": run_key, "started_at_utc": started_at_utc, **params})
		_log_dict_artifact(manifest_payload, "manifest/manifest_snapshot.yaml")

		return TrackingRunHandle(
			run_id=str(run.info.run_id),
			run_key=run_key,
			started_at_utc=started_at_utc,
			pipeline_type=pipeline_type,
			experiment_name=experiment_name,
		)


def _configure_mlflow(settings: MlflowTrackingSettings, pipeline_type: str) -> str:
	if settings.tracking_uri:
		mlflow.set_tracking_uri(settings.tracking_uri)
	experiment_name = f"{settings.experiment_prefix}_{pipeline_type}"
	mlflow.set_experiment(experiment_name)
	return experiment_name


def _resolve_manifest_name(pipeline_type: str, manifest_payload: dict[str, Any]) -> str:
	runtime_payload = manifest_payload.get("runtime")
	if isinstance(runtime_payload, dict):
		artifacts_dir = runtime_payload.get("artifacts_dir")
		if isinstance(artifacts_dir, str) and artifacts_dir.strip():
			return Path(artifacts_dir).name
	return f"{pipeline_type}_inline"


def _log_params(params: dict[str, Any]) -> None:
	for key, value in params.items():
		if value is None:
			continue
		mlflow.log_param(key, stringify_param(value))


def _log_report_payload(report_path: Path) -> None:
	parsed = parse_report_file(report_path)
	for metric_name, metric_value in parsed.metrics.items():
		mlflow.log_metric(metric_name, float(metric_value))
	if parsed.params:
		_log_params(parsed.params)


def _set_run_note_from_report(report_path: Path) -> None:
	if not report_path.exists():
		return
	content = report_path.read_text(encoding="utf-8").strip()
	if not content:
		return
	mlflow.set_tag(
		"mlflow.note.content",
		_truncate_tag_value(render_mlflow_note(content), limit=12000),
	)


def _log_profile_metrics_from_artifacts(artifact_root: Path) -> None:
	summary_path = artifact_root / "run" / "summary.yaml"
	selected_model_name: str | None = None
	if summary_path.exists():
		summary_payload = read_yaml_mapping(summary_path)
		selected_model_name = _resolve_selected_model_name(summary_payload)
		_log_selected_metrics(summary_payload)

	if selected_model_name:
		_log_selected_model_by_horizon_metrics(
			frame_path=artifact_root / "evaluation" / "metrics_by_horizon.csv",
			selected_model_name=selected_model_name,
			prefix="selected_model.by_horizon",
		)
		_log_selected_model_by_horizon_metrics(
			frame_path=artifact_root / "evaluation" / "probabilistic_metrics_by_horizon.csv",
			selected_model_name=selected_model_name,
			prefix="selected_model.probabilistic.by_horizon",
		)

	_log_forecast_by_horizon_metrics(
		frame_path=artifact_root / "forecast" / "metrics_by_horizon.csv",
		prefix="forecast.by_horizon",
	)
	_log_forecast_by_horizon_metrics(
		frame_path=artifact_root / "forecast" / "probabilistic_metrics_by_horizon.csv",
		prefix="forecast.probabilistic.by_horizon",
	)


def _log_dict_artifact(payload: dict[str, Any], artifact_file: str) -> None:
	with TemporaryDirectory(prefix="mt_mlflow_") as tmp_dir:
		target = Path(tmp_dir) / artifact_file
		write_yaml(target, payload)
		mlflow.log_artifact(str(target), artifact_path=str(Path(artifact_file).parent))


class _ModelBundlePythonModel(mlflow.pyfunc.PythonModel):
	def predict(
		self,
		context: mlflow.pyfunc.PythonModelContext,
		model_input: list[Any],
		params: dict[str, Any] | None = None,
	) -> Any:
		raise NotImplementedError("This registered model bundle is not a generic pyfunc serving endpoint")


def _log_model_bundle(model_dir: Path, artifact_path: str) -> None:
	mlflow.pyfunc.log_model(
		artifact_path=artifact_path,
		python_model=_ModelBundlePythonModel(),
		artifacts={"model_bundle": str(model_dir)},
	)


def _log_files(
	artifact_root: str | Path,
	artifact_paths: list[Path],
	artifact_prefix: str,
) -> None:
	root = Path(artifact_root)
	for artifact_path in artifact_paths:
		if not artifact_path.is_file():
			continue
		relative_parent = artifact_path.parent.relative_to(root)
		artifact_subdir = Path(artifact_prefix) / relative_parent
		mlflow.log_artifact(str(artifact_path), artifact_path=str(artifact_subdir))


def _try_register_model(
	run_id: str,
	registry_name: str,
	artifact_path: str,
	registry_tags: dict[str, str],
	registry_aliases: list[str],
	registry_description: str | None = None,
) -> None:
	try:
		client = mlflow.tracking.MlflowClient()
		_ensure_registered_model(client, registry_name)
		source = f"runs:/{run_id}/{artifact_path}"
		version = client.create_model_version(name=registry_name, source=source, run_id=run_id)
		if registry_description and hasattr(client, "update_model_version"):
			client.update_model_version(
				name=registry_name,
				version=str(version.version),
				description=registry_description,
			)
		for key, value in registry_tags.items():
			client.set_model_version_tag(registry_name, version.version, key, value)
		for alias in normalize_registry_aliases(registry_aliases):
			client.set_registered_model_alias(registry_name, alias, version.version)
		mlflow.set_tag("registry.status", "registered")
		mlflow.set_tag("registry.model_name", registry_name)
		mlflow.set_tag("registry.model_version", str(version.version))
		if registry_aliases:
			mlflow.set_tag("registry.aliases", ", ".join(normalize_registry_aliases(registry_aliases)))
	except Exception as error:
		mlflow.set_tag("registry.status", "failed")
		mlflow.set_tag("registry.model_name", registry_name)
		mlflow.set_tag("registry.error", _truncate_tag_value(str(error), limit=500))
		raise


def _ensure_registered_model(client: Any, registry_name: str) -> None:
	try:
		client.get_registered_model(registry_name)
	except Exception:
		client.create_registered_model(registry_name)


def _log_frame_row_metrics(prefix: str, frame: pd.DataFrame, metric_names: tuple[str, ...]) -> None:
	if frame.empty:
		return
	row = frame.iloc[0].to_dict()
	for metric_name in metric_names:
		metric_value = row.get(metric_name)
		if isinstance(metric_value, (int, float)):
			mlflow.log_metric(f"{prefix}.{metric_name}", float(metric_value))


def _log_frame_horizon_metrics(
	prefix: str,
	frame: pd.DataFrame,
	metric_names: tuple[str, ...],
) -> None:
	if frame.empty or "horizon" not in frame.columns:
		return
	for _, row in frame.iterrows():
		horizon = int(row["horizon"])
		for metric_name in metric_names:
			metric_value = row.get(metric_name)
			if isinstance(metric_value, (int, float)):
				mlflow.log_metric(f"{prefix}.{metric_name}", float(metric_value), step=horizon)


def _log_calibration_metrics(frame: pd.DataFrame | None) -> None:
	if frame is None or frame.empty:
		return
	mlflow.log_metric("probabilistic.calibration_rows", float(len(frame)))
	for column in ("available_errors", "radius_80", "radius_95"):
		if column not in frame.columns:
			continue
		series = pd.to_numeric(frame[column], errors="coerce").dropna()
		if series.empty:
			continue
		mlflow.log_metric(f"probabilistic.calibration_mean.{column}", float(series.mean()))
		mlflow.log_metric(f"probabilistic.calibration_max.{column}", float(series.max()))


def _log_probabilistic_metadata_metrics(payload: dict[str, Any]) -> None:
	for key, value in payload.items():
		if isinstance(value, bool):
			mlflow.log_metric(f"probabilistic.metadata.{key}", 1.0 if value else 0.0)
		elif isinstance(value, (int, float)):
			mlflow.log_metric(f"probabilistic.metadata.{key}", float(value))


def _log_selected_metrics(summary_payload: dict[str, Any]) -> None:
	selected_metrics = summary_payload.get("selected_metrics")
	if not isinstance(selected_metrics, dict):
		return
	for metric_name, metric_value in selected_metrics.items():
		if metric_name == "model_name":
			continue
		if isinstance(metric_value, (int, float)):
			mlflow.log_metric(f"selected_model.{metric_name}", float(metric_value))


def _resolve_selected_model_name(summary_payload: dict[str, Any]) -> str | None:
	value = summary_payload.get("selected_model_name")
	if isinstance(value, str) and value.strip():
		return value
	selected_metrics = summary_payload.get("selected_metrics")
	if isinstance(selected_metrics, dict):
		model_name = selected_metrics.get("model_name")
		if isinstance(model_name, str) and model_name.strip():
			return model_name
	return None


def _log_selected_model_by_horizon_metrics(
	frame_path: Path,
	selected_model_name: str,
	prefix: str,
) -> None:
	if not frame_path.exists():
		return
	frame = pd.read_csv(frame_path)
	if frame.empty or "horizon" not in frame.columns or "model_name" not in frame.columns:
		return
	selected = frame[frame["model_name"].astype(str) == selected_model_name]
	if selected.empty:
		return
	_log_horizon_series_metrics(selected, prefix=prefix)


def _log_forecast_by_horizon_metrics(frame_path: Path, prefix: str) -> None:
	if not frame_path.exists():
		return
	frame = pd.read_csv(frame_path)
	if frame.empty or "horizon" not in frame.columns:
		return
	_log_horizon_series_metrics(frame, prefix=prefix)


def _log_horizon_series_metrics(frame: pd.DataFrame, prefix: str) -> None:
	numeric_columns = [
		column
		for column in frame.columns
		if column not in {"model_name", "horizon", "scenario_name"}
		and pd.api.types.is_numeric_dtype(frame[column])
	]
	for _, row in frame.iterrows():
		horizon = int(row["horizon"])
		for column in numeric_columns:
			value = row.get(column)
			if isinstance(value, (int, float)) and pd.notna(value):
				mlflow.log_metric(f"{prefix}.{column}", float(value), step=horizon)


def _truncate_tag_value(value: str, limit: int) -> str:
	if len(value) <= limit:
		return value
	return value[: max(limit - 3, 0)] + "..."
