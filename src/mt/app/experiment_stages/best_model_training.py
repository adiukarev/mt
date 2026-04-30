from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.model_artifact.model.fitting import fit_model_bundle_from_context


class ExperimentBestModelTrainingPipelineStage(BasePipelineStage):
	def execute(self, ctx: ExperimentPipelineContext) -> None:
		selected_payload = ctx.require_selected_model_artifact_payload()

		ctx.best_model_bundle = fit_model_bundle_from_context(
			manifest=ctx.manifest,
			dataset_bundle=ctx.require_dataset(),
			feature_registry=ctx.require_feature_registry(),
			supervised=ctx.require_supervised(),
			feature_columns=ctx.feature_columns,
			model_name=selected_payload.result.info.model_name,
			model_artifact_dir=ctx.artifacts_paths_map.model,
			backtest_predictions=selected_payload.result.predictions,
			backtest_probabilistic_metadata=selected_payload.result.probabilistic_metadata,
			source_artifact=selected_payload.model_artifact,
		)
