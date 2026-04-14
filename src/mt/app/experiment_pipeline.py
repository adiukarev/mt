import pandas as pd
from pathlib import Path

from mt.app.experiment_stages.best_model_fit import BestModelFitStage
from mt.app.experiment_stages.backtest_window_generation import BacktestWindowGenerationStage
from mt.app.experiment_stages.comparison import ComparisonStage
from mt.app.experiment_stages.dataset_preparation import DatasetPreparationStage
from mt.app.experiment_stages.feature_registry import FeatureRegistryStage
from mt.app.experiment_stages.segmentation import SegmentationStage
from mt.app.experiment_stages.supervised_building import SupervisedBuildingStage
from mt.app.experiment_stages.model_execution import ModelExecutionStage
from mt.app.experiment_stages.dataset_bundling import DatasetBundlingStage
from mt.domain.experiment import ExperimentPipelineContext
from mt.domain.manifest import ExperimentManifest
from mt.domain.pipeline import BasePipeline
from mt.infra.artifact.logs.pipeline import (
	log_experiment_start,
	log_experiment_end,
	log_experiment_metrics
)
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath
from mt.infra.artifact.versioning import archive_existing_artifacts
from mt.infra.artifact.writer import write_csv, write_reports, write_plots
from mt.domain.experiment import ExperimentArtifactPathsMap


class ExperimentPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__([
			DatasetBundlingStage(),
			# preprocess start
			DatasetPreparationStage(),
			SegmentationStage(),
			FeatureRegistryStage(),
			SupervisedBuildingStage(),
			BacktestWindowGenerationStage(),
			# preprocess end
			ModelExecutionStage(),
			ComparisonStage(),
			BestModelFitStage(),
		])

	def build_context(self, manifest: ExperimentManifest) -> ExperimentPipelineContext:
		"""Подготовить контекст запуска и каталоги артефактов"""

		log_experiment_start(manifest)

		archive_existing_artifacts(manifest.runtime.artifacts_dir)

		return ExperimentPipelineContext(
			manifest=manifest,
			artifacts_paths_map=self._ensure_artifacts_paths_map(manifest.runtime.artifacts_dir),
			feature_manifest=manifest.build_combined_feature_manifest(),
			model_manifests=manifest.enabled_models,
		)

	def finalize(self, ctx: ExperimentPipelineContext) -> ExperimentPipelineContext:
		"""Сохранить финальные отчеты и вернуть заполненный контекст"""

		if ctx.dataset is None or ctx.windows is None or ctx.overall_metrics is None:
			raise ValueError()

		self._persist_artifacts(ctx)

		log_experiment_metrics(ctx)
		log_experiment_end(ctx)

		return ctx

	def _persist_artifacts(self, ctx: ExperimentPipelineContext) -> None:
		write_reports(ctx)
		write_plots(ctx)

		# каталог прогона нужен как компактный индекс всех выполненных стадий и моделей
		write_csv(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath("run_catalog.csv"),
			pd.DataFrame([
				{
					"record_type": "pipeline",
					"name": "experiment",
					"status": "completed",
					"wall_time_seconds": ctx.pipeline_wall_time_seconds,
				},
				*[
					{
						"record_type": "stage",
						"name": str(stage_info["stage_name"]),
						"status": str(stage_info["status"]),
						"wall_time_seconds": stage_info["wall_time_seconds"],
					}
					for stage_info in ctx.stage_timings
				],
				*ctx.run_catalog_rows
			]),
		)

	def _ensure_artifacts_paths_map(self, output_dit: str | Path) -> ExperimentArtifactPathsMap:
		"""Создать директории артефактов для экспериментов"""

		root = Path(output_dit)
		models_dir = root / "models"
		reports_dir = root / "run"
		tables_dir = root / "comparison"
		plots_dir = root / "plots"

		for directory in (root, models_dir, reports_dir, tables_dir, plots_dir):
			directory.mkdir(parents=True, exist_ok=True)

		return ExperimentArtifactPathsMap(
			root=root,
			models=models_dir,
			reports=reports_dir,
			tables=tables_dir,
			plots=plots_dir,
		)
