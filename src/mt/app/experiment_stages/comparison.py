import pandas as pd

from mt.domain.experiment import ExperimentPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.analysis.comparison import build_comparison_artifacts, ComparisonArtifacts
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath
from mt.infra.artifact.writer import write_csv
from mt.infra.metric.common import aggregate_metrics


class ComparisonStage(BaseStage):
	"""Этап итогового сравнения моделей"""

	name = "experiment_comparison"

	def execute(self, ctx: ExperimentPipelineContext) -> None:
		if ctx.predictions is None:
			raise ValueError()

		overall_metrics, by_horizon_metrics = (
			aggregate_metrics(ctx.predictions) if not ctx.predictions.empty else (pd.DataFrame(),
			                                                                      pd.DataFrame())
		)

		# build_comparison_artifacts собирает bootstrap CI, сегментные разрезы и error cases для анализа.
		comparison = (
			build_comparison_artifacts(
				ctx.predictions,
				seed=ctx.manifest.runtime.seed,
				bootstrap_samples=ctx.manifest.runtime.bootstrap_samples,
			)
			if not ctx.predictions.empty
			else None
		)

		# artifacts
		self._persist_artifacts(ctx, comparison, overall_metrics, by_horizon_metrics)

		ctx.overall_metrics = overall_metrics
		ctx.by_horizon_metrics = by_horizon_metrics
		ctx.comparison = comparison

	def _persist_artifacts(
		self,
		ctx: ExperimentPipelineContext,
		comparison: ComparisonArtifacts | None,
		overall_metrics: pd.DataFrame,
		by_horizon_metrics: pd.DataFrame,
	) -> None:
		if not overall_metrics.empty:
			# Сводные таблицы сравнения нужны как основной источник для выбора лучшей модели.
			write_csv(
				ctx.artifacts_paths_map.root / experiment_artifact_relpath("overall_model_comparison.csv"),
				overall_metrics
			)
			write_csv(
				ctx.artifacts_paths_map.root / experiment_artifact_relpath("metrics_by_horizon.csv"),
				by_horizon_metrics
			)

		if comparison is not None:
			write_csv(
				ctx.artifacts_paths_map.root / experiment_artifact_relpath("metrics_by_segment.csv"),
				comparison.metrics_by_segment
			)
			write_csv(
				ctx.artifacts_paths_map.root / experiment_artifact_relpath("metrics_by_category.csv"),
				comparison.metrics_by_category
			)
			write_csv(
				ctx.artifacts_paths_map.root / experiment_artifact_relpath(
					"bootstrap_ci_model_differences.csv"),
				comparison.bootstrap_ci
			)
			write_csv(
				ctx.artifacts_paths_map.root / experiment_artifact_relpath("selected_error_cases.csv"),
				comparison.error_cases
			)
			write_csv(
				ctx.artifacts_paths_map.root / experiment_artifact_relpath(
					"rolling_vs_holdout_diagnostic.csv"),
				comparison.rolling_vs_holdout
			)

			if not overall_metrics.empty:
				best_model_name = overall_metrics.iloc[0]["model_name"]
				leader_forecast = ctx.predictions[ctx.predictions["model_name"] == best_model_name].copy()
				write_csv(
					ctx.artifacts_paths_map.root / experiment_artifact_relpath("leader_forecast.csv"),
					leader_forecast
				)
