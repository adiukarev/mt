from mt.app.base_stages.raw_dataset_loading import BaseRawDatasetLoadingPipelineStage
from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext


class ExperimentRawDatasetLoadingPipelineStage(BaseRawDatasetLoadingPipelineStage):
	def execute(self, ctx: ExperimentPipelineContext) -> None:
		if ctx.raw_dataset is not None:
			return
		super().execute(ctx)
