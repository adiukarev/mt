import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.model.model_name import ModelName
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.analysis.comparison import build_evaluation_artifacts
from mt.infra.metric.aggregator import aggregate_metrics, aggregate_probabilistic_metrics
from mt.infra.model.selector import select_model_from_metrics


class ExperimentModelsEvaluationPipelineStage(BasePipelineStage):
	"""Этап итогового сравнения моделей"""

	def execute(self, ctx: ExperimentPipelineContext) -> None:
		predictions = ctx.require_predictions()

		ctx.overall_metrics, ctx.by_horizon_metrics = (
			aggregate_metrics(predictions)
			if not predictions.empty else (pd.DataFrame(), pd.DataFrame())
		)

		ctx.probabilistic_overall_metrics, ctx.probabilistic_by_horizon_metrics = (
			aggregate_probabilistic_metrics(predictions)
			if not predictions.empty else (pd.DataFrame(), pd.DataFrame())
		)

		ctx.evaluation = build_evaluation_artifacts(
			predictions,
			model_results=[payload.result for payload in ctx.model_artifact_payloads],
			seed=ctx.manifest.runtime.seed,
			bootstrap_samples=ctx.manifest.backtest.bootstrap_samples,
		)

		if not ctx.overall_metrics.empty:
			ctx.selected_model_name = select_model_from_metrics(
				ctx.overall_metrics,
				ctx.manifest.enabled_model_names,
			)
			ctx.selected_model_metrics = _build_selected_model_metrics(
				ctx.selected_model_name,
				ctx.overall_metrics,
				ctx.probabilistic_overall_metrics,
			)


def _build_selected_model_metrics(
	model_name: ModelName,
	overall_metrics: pd.DataFrame,
	probabilistic_overall_metrics: pd.DataFrame,
) -> dict[str, object]:
	result = overall_metrics.loc[
		overall_metrics["model_name"].astype(str) == model_name
	].iloc[0].to_dict()

	if (
		not probabilistic_overall_metrics.empty
		and "model_name" in probabilistic_overall_metrics.columns
	):
		selected_probabilistic = probabilistic_overall_metrics.loc[
			probabilistic_overall_metrics["model_name"].astype(str) == model_name
		]
		if not selected_probabilistic.empty:
			result.update(
				{
					key: value
					for key, value in selected_probabilistic.iloc[0].to_dict().items()
					if key != "model_name"
				}
			)
	return result
