from mt.app.audit_stages.raw_dataset_loading import AuditRawDatasetLoadingPipelineStage
from mt.app.audit_stages.dataset_preparation import AuditDatasetPreparationPipelineStage
from mt.app.audit_stages.artifact_persistence import AuditArtifactPersistencePipelineStage
from mt.app.audit_stages.dataset_segmentation import AuditDatasetSegmentationPipelineStage
from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.domain.audit.audit_pipeline_manifest import AuditPipelineManifest
from mt.domain.pipeline.pipeline import BasePipeline


class AuditPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__(
			[
				AuditRawDatasetLoadingPipelineStage(),
				AuditDatasetPreparationPipelineStage(),
				AuditDatasetSegmentationPipelineStage(),
				AuditArtifactPersistencePipelineStage(),
			]
		)

	def build_context(self, manifest: AuditPipelineManifest) -> AuditPipelineContext:
		return AuditPipelineContext(manifest=manifest)

	def finalize(self, ctx: AuditPipelineContext) -> AuditPipelineContext:
		self._persist_run_artifacts(ctx)

		return ctx
