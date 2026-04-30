from mt.app.base_stages.dataset_preparation import BaseDatasetPreparationPipelineStage
from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext


class ForecastDatasetPreparationPipelineStage(BaseDatasetPreparationPipelineStage):
	def get_dataset_manifest(self, ctx: ForecastPipelineContext) -> object:
		return ctx.require_resolved_dataset_manifest()
