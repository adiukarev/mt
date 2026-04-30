from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.domain.series_segmentation.series_segmentation import segment_series


class BaseDatasetSegmentationPipelineStage(BasePipelineStage):
	def execute(self, ctx: object) -> None:
		ctx.segments = segment_series(ctx.dataset.weekly)
