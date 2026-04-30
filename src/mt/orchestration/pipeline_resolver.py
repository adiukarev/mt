from typing import Any, Callable
from dataclasses import dataclass
from pathlib import Path

from mt.app.audit_pipeline import AuditPipeline
from mt.app.experiment_pipeline import ExperimentPipeline
from mt.app.forecast_pipeline import ForecastPipeline
from mt.app.monitoring_pipeline import MonitoringPipeline
from mt.app.synthetic_generation_pipeline import SyntheticGenerationPipeline
from mt.domain.audit.audit_pipeline_manifest import AuditPipelineManifest
from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.domain.forecast.forecast_pipeline_manifest import ForecastPipelineManifest
from mt.domain.monitoring.monitoring_pipeline_manifest import MonitoringPipelineManifest
from mt.domain.synthetic_generation.synthetic_generation_pipeline_manifest import SyntheticGenerationPipelineManifest
from mt.domain.pipeline.pipeline import BasePipeline
from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.infra.tracking.params_builder import build_tracking_params, build_tracking_tags


CLI_AND_ORCHESTRATION = "cli_and_orchestration"
ORCHESTRATION_ONLY = "orchestration_only"


@dataclass(slots=True, frozen=True)
class PipelineDefinition:
	pipeline_type: str
	pipeline_factory: Callable[[], BasePipeline]
	manifest_loader: Callable[[str | Path | dict[str, Any]], Any]
	artifact_root_resolver: Callable[[BasePipelineContext], Path]
	param_builder: Callable[[Any], dict[str, Any]]
	tag_builder: Callable[[Any], dict[str, str]]
	exposure_mode: str = CLI_AND_ORCHESTRATION
	cli_command: str | None = None
	cli_aliases: tuple[str, ...] = ()
	cli_help: str | None = None


def resolve_pipeline_definition(pipeline_type: str) -> PipelineDefinition:
	try:
		return resolve_pipeline_definitions()[pipeline_type.strip().lower()]
	except KeyError as error:
		raise KeyError() from error


def resolve_pipeline_definitions() -> dict[str, PipelineDefinition]:
	return {
		"audit": PipelineDefinition(
			pipeline_type="audit",
			pipeline_factory=AuditPipeline,
			manifest_loader=AuditPipelineManifest.load,
			artifact_root_resolver=_artifact_root_from_context,
			param_builder=_build_audit_tracking_params,
			tag_builder=_build_audit_tracking_tags,
			cli_command="audit",
			cli_help="Запустить аудит данных",
		),
		"experiment": PipelineDefinition(
			pipeline_type="experiment",
			pipeline_factory=ExperimentPipeline,
			manifest_loader=ExperimentPipelineManifest.load,
			artifact_root_resolver=_artifact_root_from_context,
			param_builder=_build_experiment_tracking_params,
			tag_builder=_build_experiment_tracking_tags,
			cli_command="experiment",
			cli_help="Запустить эксперимент",
		),
		"monitoring": PipelineDefinition(
			pipeline_type="monitoring",
			pipeline_factory=MonitoringPipeline,
			manifest_loader=MonitoringPipelineManifest.load,
			artifact_root_resolver=_artifact_root_from_context,
			param_builder=_build_monitoring_tracking_params,
			tag_builder=_build_monitoring_tracking_tags,
			cli_command="monitoring",
			cli_help="Запустить monitoring synthetic source без downstream trigger",
		),
		"synthetic_generation": PipelineDefinition(
			pipeline_type="synthetic_generation",
			pipeline_factory=SyntheticGenerationPipeline,
			manifest_loader=SyntheticGenerationPipelineManifest.load,
			artifact_root_resolver=_artifact_root_from_context,
			param_builder=_build_synthetic_tracking_params,
			tag_builder=_build_synthetic_tracking_tags,
			cli_command="synthetic-generation",
			cli_help="Сгенерировать synthetic weekly retail-датасет",
		),
		"forecast": PipelineDefinition(
			pipeline_type="forecast",
			pipeline_factory=ForecastPipeline,
			manifest_loader=ForecastPipelineManifest.load,
			artifact_root_resolver=_artifact_root_from_context,
			param_builder=_build_forecast_tracking_params,
			tag_builder=_build_forecast_tracking_tags,
			cli_command="forecast",
			cli_help="Построить прогноз по weekly dataset",
		),
	}


def resolve_cli_pipeline_definitions() -> dict[str, PipelineDefinition]:
	command_mapping: dict[str, PipelineDefinition] = {}
	for definition in resolve_pipeline_definitions().values():
		if definition.exposure_mode != CLI_AND_ORCHESTRATION:
			continue
		if definition.cli_command is None:
			continue
		command_mapping[definition.cli_command] = definition
		for alias in definition.cli_aliases:
			command_mapping[alias] = definition
	return command_mapping


def _build_experiment_tracking_params(manifest: Any) -> dict[str, Any]:
	return build_tracking_params("experiment", manifest)


def _build_experiment_tracking_tags(manifest: Any) -> dict[str, str]:
	return build_tracking_tags("experiment", manifest)


def _build_audit_tracking_params(manifest: Any) -> dict[str, Any]:
	return build_tracking_params("audit", manifest)


def _build_audit_tracking_tags(manifest: Any) -> dict[str, str]:
	return build_tracking_tags("audit", manifest)


def _build_synthetic_tracking_params(manifest: Any) -> dict[str, Any]:
	return build_tracking_params("synthetic_generation", manifest)


def _build_synthetic_tracking_tags(manifest: Any) -> dict[str, str]:
	return build_tracking_tags("synthetic_generation", manifest)


def _build_forecast_tracking_params(manifest: Any) -> dict[str, Any]:
	return build_tracking_params("forecast", manifest)


def _build_forecast_tracking_tags(manifest: Any) -> dict[str, str]:
	return build_tracking_tags("forecast", manifest)


def _build_monitoring_tracking_params(manifest: Any) -> dict[str, Any]:
	return build_tracking_params("monitoring", manifest)


def _build_monitoring_tracking_tags(manifest: Any) -> dict[str, str]:
	return build_tracking_tags("monitoring", manifest)


def _artifact_root_from_context(ctx: BasePipelineContext) -> Path:
	return ctx.artifacts_paths_map.root
