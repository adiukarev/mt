from mt.app.synthetic_generation_stages.dataset_generation import (
	SyntheticGenerationDatasetGenerationPipelineStage,
)
from mt.app.synthetic_generation_stages.artifact_persistence import (
	SyntheticGenerationArtifactPersistencePipelineStage,
)
from mt.domain.pipeline.pipeline import BasePipeline
from mt.domain.synthetic_generation.synthetic_generation_pipeline_manifest import SyntheticGenerationPipelineManifest
from mt.domain.synthetic_generation.synthetic_generation_pipeline_context import SyntheticGenerationPipelineContext


class SyntheticGenerationPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__([
			SyntheticGenerationDatasetGenerationPipelineStage(),
			SyntheticGenerationArtifactPersistencePipelineStage(),
		])

	def build_context(
		self,
		manifest: SyntheticGenerationPipelineManifest
	) -> SyntheticGenerationPipelineContext:
		return SyntheticGenerationPipelineContext(manifest=manifest)

	def finalize(self, ctx: SyntheticGenerationPipelineContext) -> SyntheticGenerationPipelineContext:
		self._persist_run_artifacts(ctx)

		return ctx
