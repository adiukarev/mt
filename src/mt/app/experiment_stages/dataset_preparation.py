from mt.domain.experiment import ExperimentPipelineContext
from mt.app.base_stages.dataset_preparation import BaseDatasetPreparationStage
from mt.infra.artifact.writer import write_markdown
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath


class DatasetPreparationStage(BaseDatasetPreparationStage):
	name = "experiment_dataset_preparation"

	def get_dataset_manifest(self, ctx: ExperimentPipelineContext):
		return ctx.manifest.dataset

	def after_execute(self, ctx: ExperimentPipelineContext) -> None:
		self._persist_artifacts(ctx)

	def _persist_artifacts(self, ctx: ExperimentPipelineContext) -> None:
		write_markdown(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath("dataset_preparation_summary.md"),
			[
				"# Подготовка датасета",
				"",
				f"- уровень агрегации: {ctx.dataset.aggregation_level}",
				f"- число недельных строк: {len(ctx.dataset.weekly)}",
				f"- число недельных рядов: {ctx.dataset.weekly['series_id'].nunique()}",
				"",
				f"- series_limit: {ctx.manifest.dataset.series_limit if ctx.manifest.dataset.series_limit is not None else 'all'}",
			],
		)
