import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from uuid import uuid4

from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.orchestration.orchestration_artifact import OrchestrationArtifactPathsMap
from mt.infra.runtime.runtime import (
	ensure_runtime_env,
	ensure_runtime_logging,
	ensure_runtime_seed_everything,
)
from mt.infra.observability.runtime.bootstrap import initialize_observability, attach_observability
from mt.infra.stage_io.artifact_inventory import (
	build_artifact_inventory,
	changed_artifact_paths,
	tracked_artifact_paths,
)
from mt.infra.stage_io.context_snapshot import load_context_snapshot, save_context_snapshot
from mt.infra.tracking.backend_builder import build_tracking_backend
from mt.infra.tracking.backend_resolver import resolve_tracking_backend_name
from mt.infra.tracking.metrics_builder import build_final_metrics
from mt.orchestration.monitoring_experiment_trigger import (
	build_monitoring_experiment_manifest,
)
from mt.orchestration.pipeline_resolver import resolve_pipeline_definition


def initialize_pipeline_run(
	pipeline_type: str,
	manifest_payload: dict[str, Any],
	dag_id: str | None = None,
) -> dict[str, Any]:
	"""Подготовить context snapshot и parent tracking run для Airflow/Docker orchestration"""

	os.environ["MT_EXECUTION_MODE"] = "airflow"
	ensure_runtime_env()
	ensure_runtime_logging()

	definition = resolve_pipeline_definition(pipeline_type)
	manifest = definition.manifest_loader(manifest_payload)
	resolved_manifest_payload = manifest.to_dict()
	tracking_params = definition.param_builder(manifest)
	tracking_tags = definition.tag_builder(manifest)
	artifact_root, orchestration_paths, logical_artifact_root = _resolve_runtime_paths(
		pipeline_type=pipeline_type,
		requested_artifact_root=Path(str(manifest.runtime.artifacts_dir)),
	)
	manifest.runtime.artifacts_dir = str(artifact_root)
	seed = getattr(getattr(manifest, "runtime", None), "seed", None)
	if isinstance(seed, int):
		ensure_runtime_seed_everything(seed)

	pipeline = definition.pipeline_factory()
	ctx = pipeline.build_context(manifest=manifest)
	ctx.runtime_metadata["logical_manifest_payload"] = resolved_manifest_payload
	ctx.runtime_metadata["logical_artifact_root"] = str(logical_artifact_root)
	ctx.runtime_metadata["scratch_artifact_root"] = str(artifact_root)
	initialize_observability(
		ctx=ctx,
		pipeline_type=pipeline_type,
		manifest=resolved_manifest_payload,
		artifacts_dir=logical_artifact_root,
		runtime_log_path=None,
		events_path=None,
		execution_mode="airflow",
		tracking_tags=tracking_tags,
		tracking_params=tracking_params,
	)

	ctx.runtime_metadata.update(
		{
			"pipeline_type": pipeline_type,
			"airflow_dag_id": dag_id,
			"artifact_root": str(artifact_root),
			"orchestration_dir": str(orchestration_paths.root),
			"orchestration_snapshots_dir": str(orchestration_paths.context_snapshots),
			"orchestration_stage_states_dir": str(orchestration_paths.stage_states),
			"pipeline_started_at_unix": time.time(),
			"mlflow_parent_run_id": ctx.observability.tracking_run_id if ctx.observability else None,
			"tracking_run_key": ctx.observability.run_key if ctx.observability else None,
			"tracking_started_at_utc": ctx.observability.started_at_utc if ctx.observability else None,
			"tracking_namespace": (
				ctx.observability.tracking_namespace if ctx.observability else None
			),
		}
	)

	snapshot_path = _build_snapshot_path(orchestration_paths.context_snapshots, 0,
	                                     "context_initialized")
	save_context_snapshot(
		snapshot_path,
		ctx,
		metadata=None,
	)

	return {
		"pipeline_type": pipeline_type,
		"artifact_root": str(artifact_root),
		"logical_artifact_root": str(logical_artifact_root),
		"snapshot_path": str(snapshot_path),
		"mlflow_parent_run_id": ctx.observability.tracking_run_id if ctx.observability else None,
		"tracking_run_key": ctx.observability.run_key if ctx.observability else None,
		"executed_stage_count": 0,
	}


def execute_pipeline_stage(state: dict[str, Any], stage_name: str) -> dict[str, Any]:
	"""Загрузить snapshot, выполнить один stage и сохранить новый snapshot"""

	ensure_runtime_env()
	ensure_runtime_logging()

	ctx = load_context_snapshot(state["snapshot_path"])
	attach_observability(ctx)
	pipeline_type = str(state["pipeline_type"])
	definition = resolve_pipeline_definition(pipeline_type)
	pipeline = definition.pipeline_factory()
	artifact_root = Path(str(state["artifact_root"]))
	stage_artifact_paths: list[Path] = []
	should_log_stage_artifacts = _should_log_stage_artifacts()
	if should_log_stage_artifacts:
		before_inventory = build_artifact_inventory(artifact_root)

	pipeline.run_stage(ctx, stage_name)

	if should_log_stage_artifacts:
		after_inventory = build_artifact_inventory(artifact_root)
		stage_artifact_paths = changed_artifact_paths(artifact_root, before_inventory, after_inventory)

	stage_index = int(state["executed_stage_count"]) + 1
	snapshot_path = _build_snapshot_path(
		Path(str(ctx.runtime_metadata["orchestration_snapshots_dir"])),
		stage_index,
		stage_name,
	)
	save_context_snapshot(
		snapshot_path,
		ctx,
		metadata=None,
	)

	build_tracking_backend(
		resolve_tracking_backend_name(
			ctx.observability.execution_mode if ctx.observability else None
		)
	).log_stage_run(
		run_id=_parent_run_id(ctx),
		stage_name=stage_name,
		stage_timing=ctx.stage_timings[-1] if ctx.stage_timings else None,
		artifact_paths=stage_artifact_paths,
		artifact_root=artifact_root,
	)

	return {
		**state,
		"snapshot_path": str(snapshot_path),
		"executed_stage_count": stage_index,
		"last_stage_name": stage_name,
	}


def finalize_pipeline_run(state: dict[str, Any]) -> dict[str, Any]:
	"""Выполнить pipeline.finalize над последним snapshot и сохранить финальное состояние."""

	ensure_runtime_env()
	ensure_runtime_logging()

	ctx = load_context_snapshot(state["snapshot_path"])
	attach_observability(ctx)
	pipeline_type = str(state["pipeline_type"])
	definition = resolve_pipeline_definition(pipeline_type)
	pipeline = definition.pipeline_factory()
	artifact_root = Path(str(state["artifact_root"]))

	started_at = ctx.runtime_metadata.get("pipeline_started_at_unix")
	if isinstance(started_at, (int, float)):
		ctx.pipeline_wall_time_seconds = time.time() - float(started_at)

	pipeline.finalize(ctx)

	final_artifact_paths = tracked_artifact_paths(artifact_root)

	stage_index = int(state["executed_stage_count"]) + 1
	snapshot_path = _build_snapshot_path(
		Path(str(ctx.runtime_metadata["orchestration_snapshots_dir"])),
		stage_index,
		"pipeline_finalized",
	)
	save_context_snapshot(
		snapshot_path,
		ctx,
		metadata=None,
	)

	build_tracking_backend(
		resolve_tracking_backend_name(
			ctx.observability.execution_mode if ctx.observability else None
		)
	).log_pipeline_summary(
		run_id=_parent_run_id(ctx),
		metrics=_final_metrics(ctx),
		artifact_paths=final_artifact_paths,
		artifact_root=artifact_root,
	)
	_cleanup_airflow_scratch_root(Path(str(state["artifact_root"])))

	return {
		**state,
		"snapshot_path": str(snapshot_path),
		"executed_stage_count": stage_index,
		"monitoring_experiment_manifest": _build_monitoring_experiment_manifest_payload(
			pipeline_type=pipeline_type,
			ctx=ctx,
		),
		"status": "completed",
	}


def load_pipeline_context(snapshot_path: str | Path) -> BasePipelineContext:
	"""Утилита для тестов и локальной диагностики orchestration snapshots."""

	return load_context_snapshot(snapshot_path)


def _build_snapshot_path(snapshots_dir: Path, index: int, stage_name: str) -> Path:
	return snapshots_dir / f"{index:03d}_{stage_name}.pkl.gz"


def _resolve_runtime_paths(
	pipeline_type: str,
	requested_artifact_root: Path,
) -> tuple[Path, OrchestrationArtifactPathsMap, Path]:
	logical_artifact_root = requested_artifact_root
	artifact_root = requested_artifact_root
	if _should_use_ephemeral_airflow_artifacts():
		artifact_root = _build_airflow_scratch_root(
			pipeline_type=pipeline_type,
			logical_artifact_root=logical_artifact_root,
		)
	orchestration_paths = OrchestrationArtifactPathsMap.ensure(artifact_root)
	return artifact_root, orchestration_paths, logical_artifact_root


def _build_airflow_scratch_root(pipeline_type: str, logical_artifact_root: Path) -> Path:
	slug = logical_artifact_root.name or pipeline_type
	root = (
		Path(tempfile.gettempdir())
		/ "mt_airflow_runs"
		/ pipeline_type
		/ f"{slug}_{uuid4().hex[:12]}"
	)
	root.mkdir(parents=True, exist_ok=True)
	return root


def _should_use_ephemeral_airflow_artifacts() -> bool:
	flag = os.getenv("MT_AIRFLOW_PERSIST_LOCAL_ARTIFACTS", "0").strip().lower()
	return flag not in {"1", "true", "yes"}


def _should_log_stage_artifacts() -> bool:
	flag = os.getenv("MT_AIRFLOW_LOG_STAGE_ARTIFACTS", "0").strip().lower()
	return flag in {"1", "true", "yes"}


def _cleanup_airflow_scratch_root(artifact_root: Path) -> None:
	if not _should_use_ephemeral_airflow_artifacts():
		return
	temp_root = Path(tempfile.gettempdir()).resolve()
	resolved_artifact_root = artifact_root.resolve()
	if temp_root not in resolved_artifact_root.parents:
		return
	shutil.rmtree(resolved_artifact_root, ignore_errors=True)


def _build_monitoring_experiment_manifest_payload(
	pipeline_type: str,
	ctx: BasePipelineContext,
) -> dict[str, object] | None:
	if pipeline_type != "monitoring" or not isinstance(ctx, MonitoringPipelineContext):
		return None
	return build_monitoring_experiment_manifest(ctx)


def _parent_run_id(ctx: BasePipelineContext) -> str | None:
	run_id = ctx.runtime_metadata.get("mlflow_parent_run_id")
	return str(run_id) if run_id else None


def _final_metrics(ctx: BasePipelineContext) -> dict[str, float]:
	return build_final_metrics(ctx)
