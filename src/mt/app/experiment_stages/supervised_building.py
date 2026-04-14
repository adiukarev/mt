from mt.app.base_stages.supervised_building import BaseSupervisedBuildingStage
from mt.domain.experiment import ExperimentPipelineContext
from mt.infra.artifact.writer import write_markdown
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath


class SupervisedBuildingStage(BaseSupervisedBuildingStage):
	name = "experiment_supervised_building"

	def after_execute(self, ctx: ExperimentPipelineContext) -> None:
		write_markdown(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath(f"{self.name}.md"),
			[
				"# Построение supervised-таблицы",
				"",
				f"- набор всех признаков: {ctx.feature_manifest.feature_set}",
				"- помодельное подмножество колонок разрешается отдельно на этапе model_execution",
				f"- колонки признаков: {len(ctx.feature_columns)}",
				f"- количество строк: {len(ctx.supervised) if ctx.supervised is not None else 0}",
				f"- целевые горизонты: {ctx.manifest.backtest.horizon_min}..{ctx.manifest.backtest.horizon_max}",
			]
		)
