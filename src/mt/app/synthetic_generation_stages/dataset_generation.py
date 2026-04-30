from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.domain.synthetic_generation.synthetic_generation_pipeline_context import SyntheticGenerationPipelineContext
from mt.infra.synthetic.materializer import materialize_synthetic_dataset


class SyntheticGenerationDatasetGenerationPipelineStage(BasePipelineStage):
	def execute(self, ctx: SyntheticGenerationPipelineContext) -> None:
		result = materialize_synthetic_dataset(
			manifest=ctx.manifest,
			output_root=ctx.manifest.output.dataset_dir,
		)
		ctx.dataset = result.full_dataset
		ctx.metadata = result.series_metadata
		ctx.dataset_root = str(result.output_root)
		ctx.materialized_paths = result.materialized_paths
		ctx.generation_summary = result.summary
