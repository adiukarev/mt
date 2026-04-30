from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.forecast_artifact.writer import write_forecast_artifacts


class ForecastArtifactPersistencePipelineStage(BasePipelineStage):
	def execute(self, ctx: ForecastPipelineContext) -> None:
		write_forecast_artifacts(ctx)
