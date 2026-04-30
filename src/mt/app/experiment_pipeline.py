from mt.app.experiment_stages.backtest_window_preparation import (
	ExperimentBacktestWindowPreparationPipelineStage
)
from mt.app.experiment_stages.best_model_training import (
	ExperimentBestModelTrainingPipelineStage,
)
from mt.app.experiment_stages.models_evaluation import ExperimentModelsEvaluationPipelineStage
from mt.app.experiment_stages.data_source_sync import ExperimentDataSourceSyncPipelineStage
from mt.app.experiment_stages.dataset_preparation import ExperimentDatasetPreparationPipelineStage
from mt.app.experiment_stages.feature_engineering import ExperimentFeatureEngineeringPipelineStage
from mt.app.experiment_stages.dataset_segmentation import ExperimentDatasetSegmentationPipelineStage
from mt.app.experiment_stages.supervised_building import ExperimentSupervisedBuildingPipelineStage
from mt.app.experiment_stages.models_training import ExperimentModelsTrainingPipelineStage
from mt.app.experiment_stages.raw_dataset_loading import ExperimentRawDatasetLoadingPipelineStage
from mt.app.experiment_stages.artifact_persistence import ExperimentArtifactPersistencePipelineStage
from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.domain.pipeline.pipeline import BasePipeline
from mt.infra.observability.runtime.stage_events import (
	log_experiment_start,
	log_experiment_end,
	log_experiment_metrics
)


class ExperimentPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__([
			ExperimentDataSourceSyncPipelineStage(),
			ExperimentRawDatasetLoadingPipelineStage(),
			ExperimentDatasetPreparationPipelineStage(),
			ExperimentDatasetSegmentationPipelineStage(),
			ExperimentFeatureEngineeringPipelineStage(),
			ExperimentSupervisedBuildingPipelineStage(),
			ExperimentBacktestWindowPreparationPipelineStage(),
			ExperimentModelsTrainingPipelineStage(),
			ExperimentModelsEvaluationPipelineStage(),
			ExperimentBestModelTrainingPipelineStage(),
			ExperimentArtifactPersistencePipelineStage(),
		])

	def build_context(self, manifest: ExperimentPipelineManifest) -> ExperimentPipelineContext:
		"""Подготовить контекст запуска и каталоги артефактов"""

		log_experiment_start(manifest)

		return ExperimentPipelineContext(
			manifest=manifest,
			feature_manifest=manifest.build_combined_feature_manifest(),
			model_manifests=manifest.enabled_models,
		)

	def finalize(self, ctx: ExperimentPipelineContext) -> ExperimentPipelineContext:
		"""Сохранить финальные отчеты и вернуть заполненный контекст"""

		self._persist_run_artifacts(ctx, extra_rows=ctx.run_catalog_rows)

		log_experiment_metrics(ctx)
		log_experiment_end(ctx)

		return ctx
