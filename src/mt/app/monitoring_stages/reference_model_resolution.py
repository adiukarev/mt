from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.forecast.reference_model import load_reference_model_config
from mt.infra.observability.logger.runtime_logger import log_warning


class MonitoringReferenceModelResolutionPipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		try:
			execution_mode = ctx.observability.execution_mode if ctx.observability else None
			ctx.champion_model = load_reference_model_config(
				model_manifest=ctx.manifest.model,
				execution_mode=execution_mode,
			)
			ctx.source_descriptor.update(
				{
					f"model_{key}": value
					for key, value in (ctx.champion_model.source_descriptor or {}).items()
				}
			)
			ctx.source_descriptor["champion_model_name"] = ctx.champion_model.model_name.value
		except Exception as error:
			ctx.monitoring_issues.append("champion_model_unavailable")
			ctx.source_descriptor["champion_model_error"] = str(error)
			log_warning(
				"monitoring",
				scope="monitoring",
				reason="champion_model_unavailable",
				error=str(error),
			)
