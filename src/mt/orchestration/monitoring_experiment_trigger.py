from dataclasses import asdict
from pathlib import Path
from typing import Any

from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.infra.dataset.sources.source_resolver import infer_dataset_source_manifest


DEFAULT_MONITORING_EXPERIMENT_MANIFEST_PATH = "manifests/monitoring_experiment.yaml"


def build_monitoring_experiment_trigger_conf(
	state: dict[str, object],
) -> dict[str, object] | None:
	manifest = dict(state.get("monitoring_experiment_manifest") or {})
	if not manifest:
		return None
	return {"manifest": manifest}


def build_monitoring_experiment_manifest(
	ctx: MonitoringPipelineContext,
	template_manifest_path: str = DEFAULT_MONITORING_EXPERIMENT_MANIFEST_PATH,
) -> dict[str, Any] | None:
	if ctx.decision_artifact is None or not ctx.decision_artifact.should_run_experiment:
		return None

	template_manifest = ExperimentPipelineManifest.load(template_manifest_path).to_dict()
	dataset_payload = asdict(ctx.manifest.dataset)
	dataset_payload["path"] = str(
		ctx.source_descriptor.get("dataset_path") or ctx.manifest.dataset.path
	)
	template_manifest["dataset"] = dataset_payload
	template_manifest["source"] = _build_source_payload(ctx)
	template_manifest["runtime"]["seed"] = ctx.manifest.runtime.seed
	template_manifest["runtime"]["artifacts_dir"] = str(
		_derive_experiment_artifacts_dir(ctx.manifest.runtime.artifacts_dir)
	)
	return template_manifest


def _build_source_payload(
	ctx: MonitoringPipelineContext,
) -> dict[str, Any] | None:
	source_manifest = infer_dataset_source_manifest(ctx.manifest.dataset)
	source_payload = asdict(source_manifest)
	source_type = str(source_manifest.source_type.value).strip().lower()
	if source_type in {"", "local_files"}:
		return None
	return source_payload


def _derive_experiment_artifacts_dir(monitoring_artifacts_dir: str) -> Path:
	monitoring_root = Path(monitoring_artifacts_dir)
	if "monitoring" in monitoring_root.name:
		return monitoring_root.parent / "experiment_refresh"
	return monitoring_root.parent / f"{monitoring_root.name}_experiment_refresh"
