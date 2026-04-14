from mt.domain.audit import AuditPipelineContext
from mt.app.base_stages.dataset_bundling import BaseDatasetBundlingStage


class AuditDatasetBundlingStage(BaseDatasetBundlingStage):
	name = "audit_dataset_bundling"

	def get_dataset_manifest(self, ctx: AuditPipelineContext):
		return ctx.dataset_manifest
