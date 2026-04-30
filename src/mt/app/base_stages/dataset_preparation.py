from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.dataset.preparation import prepare_dataset


class BaseDatasetPreparationPipelineStage(BasePipelineStage):
	"""Общий шаг превращения raw-таблиц в недельный dataset"""

	def execute(self, ctx: object) -> None:
		ctx.dataset = prepare_dataset(self.get_dataset_manifest(ctx), ctx.raw_dataset)

		post_prepare = getattr(self, "post_prepare", None)
		if callable(post_prepare):
			post_prepare(ctx)

	def get_dataset_manifest(self, ctx: object):
		return ctx.dataset_manifest
