from pathlib import Path

from mt.app.audit_stages.dataset_bundling import AuditDatasetBundlingStage
from mt.app.audit_stages.dataset_preparation import AuditDatasetPreparationStage
from mt.app.audit_stages.persist_artifacts import AuditPersistArtifactsStage
from mt.app.audit_stages.segmentation import AuditSegmentationStage
from mt.domain.audit import AuditPipelineContext
from mt.domain.manifest import AuditManifest
from mt.domain.pipeline import BasePipeline
from mt.infra.artifact.versioning import archive_existing_artifacts


class AuditPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__(
			[
				AuditDatasetBundlingStage(),
				AuditDatasetPreparationStage(),
				AuditSegmentationStage(),
				AuditPersistArtifactsStage(),
			]
		)

	def build_context(self, manifest: AuditManifest) -> AuditPipelineContext:
		output_dir = Path(manifest.runtime.output_dir)

		archive_existing_artifacts(output_dir)

		return AuditPipelineContext(
			dataset_manifest=manifest.dataset,
			output_dir=output_dir,
		)

	def finalize(self, ctx: AuditPipelineContext) -> AuditPipelineContext:
		return ctx
