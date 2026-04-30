from mt.app.base_stages.dataset_preparation import BaseDatasetPreparationPipelineStage
from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.infra.dataset.factory import build_dataset_adapter


class AuditDatasetPreparationPipelineStage(BaseDatasetPreparationPipelineStage):
	def post_prepare(self, ctx: AuditPipelineContext) -> None:
		if ctx.raw_dataset is None or ctx.dataset is None:
			raise ValueError()

		ctx.raw_context = build_dataset_adapter(ctx.dataset_manifest).build_raw_context(
			ctx.raw_dataset,
			ctx.dataset,
		)
