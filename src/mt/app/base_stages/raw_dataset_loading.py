from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.dataset.loader import load_dataset


class BaseRawDatasetLoadingPipelineStage(BasePipelineStage):
	def execute(self, ctx: object) -> None:
		ctx.raw_dataset = load_dataset(self.get_dataset_manifest(ctx))

	def get_dataset_manifest(self, ctx: object):
		return ctx.dataset_manifest
