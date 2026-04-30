from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
from typing import Any

from mt.domain.observability.observability import ObservabilityContext
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.tracking.tracking_request import TrackingRequest
from mt.infra.artifact.text_writer import write_yaml
from mt.infra.observability.logger.runtime_logger import configure_runtime_logging
from mt.infra.observability.runtime.context_store import bind_observability
from mt.infra.observability.runtime.event_emitter import emit_event
from mt.infra.tracking.backend_builder import build_tracking_backend
from mt.infra.tracking.backend_resolver import resolve_tracking_backend_name


def initialize_observability(
	ctx: BasePipelineContext,
	pipeline_type: str,
	manifest: object,
	artifacts_dir: str | Path,
	runtime_log_path: str | Path | None = None,
	events_path: str | Path | None = None,
	execution_mode: str | None = None,
	tracking_tags: dict[str, str] | None = None,
	tracking_params: dict[str, Any] | None = None,
) -> BasePipelineContext:
	manifest_payload = _manifest_payload(manifest)
	resolved_execution_mode = execution_mode or os.getenv("MT_EXECUTION_MODE", "local")
	backend_name = resolve_tracking_backend_name(resolved_execution_mode)
	backend = build_tracking_backend(backend_name)
	tracking_handle = backend.start_run(
		TrackingRequest(
			pipeline_type=pipeline_type,
			manifest_payload=manifest_payload,
			params=tracking_params or {},
			tags=tracking_tags,
		)
	)

	observability = ObservabilityContext(
		run_key=tracking_handle.run_key,
		pipeline_type=pipeline_type,
		artifacts_dir=str(artifacts_dir),
		started_at_utc=tracking_handle.started_at_utc,
		execution_mode=resolved_execution_mode,
		tracking_run_id=tracking_handle.run_id,
		tracking_namespace=tracking_handle.experiment_name,
		runtime_log_path=str(runtime_log_path) if runtime_log_path is not None else None,
		events_path=str(events_path) if events_path is not None else None,
	)
	ctx.observability = observability
	ctx.runtime_metadata.update(
		{
			"tracking_backend": backend.name,
			"tracking_run_key": observability.run_key,
			"mlflow_parent_run_id": observability.tracking_run_id,
			"tracking_namespace": observability.tracking_namespace,
			"execution_mode": observability.execution_mode,
		}
	)

	configure_runtime_logging(runtime_log_path=runtime_log_path)
	bind_observability(observability)
	_persist_tracking_snapshot(observability)
	emit_event("pipeline_started", {"artifacts_dir": str(artifacts_dir)})
	return ctx


def attach_observability(ctx: BasePipelineContext) -> None:
	if ctx.observability is None:
		return
	configure_runtime_logging(runtime_log_path=ctx.observability.runtime_log_path)
	bind_observability(ctx.observability)


def emit_pipeline_completed(ctx: BasePipelineContext) -> None:
	if ctx.observability is None:
		return
	emit_event(
		"pipeline_completed",
		{
			"pipeline_wall_time_seconds": ctx.pipeline_wall_time_seconds,
			"executed_stages": list(ctx.executed_stages),
		},
	)


def _persist_tracking_snapshot(observability: ObservabilityContext) -> None:
	if observability.runtime_log_path is None:
		return
	orchestration_dir = Path(observability.runtime_log_path).parent
	write_yaml(
		orchestration_dir / "tracking_snapshot.yaml",
		{
			"run_key": observability.run_key,
			"pipeline_type": observability.pipeline_type,
			"tracking_backend": resolve_tracking_backend_name(observability.execution_mode),
			"tracking_run_id": observability.tracking_run_id,
			"tracking_namespace": observability.tracking_namespace,
			"execution_mode": observability.execution_mode,
		},
	)


def _manifest_payload(manifest: object) -> dict[str, Any]:
	if hasattr(manifest, "to_dict"):
		return getattr(manifest, "to_dict")()
	if is_dataclass(manifest):
		return asdict(manifest)
	if isinstance(manifest, dict):
		return manifest
	return {}
