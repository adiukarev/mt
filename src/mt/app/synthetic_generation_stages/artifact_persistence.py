from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.domain.synthetic_generation.synthetic_generation_pipeline_context import SyntheticGenerationPipelineContext
from mt.infra.synthetic_generation_artifact.writer import write_synthetic_generation_artifacts


class SyntheticGenerationArtifactPersistencePipelineStage(BasePipelineStage):
	def execute(self, ctx: SyntheticGenerationPipelineContext) -> None:
		write_synthetic_generation_artifacts(ctx)
