from mt.app.base_stages.feature_registry import BaseFeatureRegistryStage
from mt.domain.experiment import ExperimentPipelineContext
from mt.infra.artifact.writer import write_csv, write_markdown
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath


class FeatureRegistryStage(BaseFeatureRegistryStage):
	name = "experiment_feature_registry"

	def after_execute(self, ctx: ExperimentPipelineContext) -> None:
		if ctx.feature_registry is None:
			raise ValueError()

		write_csv(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath("feature_registry.csv"),
			ctx.feature_registry
		)
		write_markdown(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath(f"{self.name}.md"),
			[
				"# Реестр признаков",
				"",
				f"- набор всех признаков: {ctx.feature_manifest.feature_set}",
				f"- число строк в реестре: {len(ctx.feature_registry)}",
				"- назначение: методически задокументировать все допустимые признаки общего supervised-слоя",
				"- фактическое использование по моделям сохраняется отдельно в `model_feature_usage.csv`",
			]
		)
