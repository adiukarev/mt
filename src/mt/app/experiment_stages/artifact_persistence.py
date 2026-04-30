from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.experiment_artifact.writer import write_experiment_artifacts


class ExperimentArtifactPersistencePipelineStage(BasePipelineStage):
	def execute(self, ctx: ExperimentPipelineContext) -> None:
		ctx.require_dataset()
		ctx.require_windows()
		ctx.require_overall_metrics()
		ctx.require_best_model_bundle()
		write_experiment_artifacts(ctx)
