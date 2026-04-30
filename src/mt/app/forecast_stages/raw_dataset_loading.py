from mt.app.base_stages.raw_dataset_loading import BaseRawDatasetLoadingPipelineStage
from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext


class ForecastRawDatasetLoadingPipelineStage(BaseRawDatasetLoadingPipelineStage):
	def get_dataset_manifest(self, ctx: ForecastPipelineContext) -> object:
		return ctx.require_resolved_dataset_manifest()
