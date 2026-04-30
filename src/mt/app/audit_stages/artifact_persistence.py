from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.audit.data_builder import build_data_audit
from mt.infra.audit_artifact.writer import write_audit_artifacts


class AuditArtifactPersistencePipelineStage(BasePipelineStage):
	def execute(self, ctx: AuditPipelineContext) -> None:
		if ctx.dataset is None or ctx.segments is None:
			raise ValueError()

		ctx.audit_artifacts = build_data_audit(
			ctx.dataset.weekly,
			ctx.segments,
			ctx.dataset.metadata,
			ctx.dataset.aggregation_level,
			ctx.raw_context,
		)

		write_audit_artifacts(ctx)
