import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.experiment.experiment_artifact import ExperimentModelArtifactPayload
from mt.domain.model.model_manifest import ModelManifest
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.observability.runtime.stage_events import log_model_runner_start, log_model_runner_end
from mt.infra.metric.aggregator import aggregate_metrics, aggregate_probabilistic_metrics
from mt.infra.model.feature_resolver import resolve_model_feature_columns
from mt.infra.backtest.runner import run_backtest


class ExperimentModelsTrainingPipelineStage(BasePipelineStage):
	"""Этап запуска моделей и сохранения модельных артефактов"""

	def execute(self, ctx: ExperimentPipelineContext) -> None:
		"""Прогнать все модели из манифеста и собрать общие прогнозы"""

		all_predictions: list[pd.DataFrame] = []
		model_catalog_rows: list[dict[str, object]] = []
		model_feature_usage_rows: list[dict[str, object]] = []

		for model_manifest in ctx.model_manifests:
			model_name = model_manifest.name

			log_model_runner_start(model_name, ctx.windows)

			# supervised общий, окна тоже общие
			# отличие между моделями тут только в model_feature_columns и config
			result = run_backtest(
				model_name,
				ctx.supervised,
				self._resolve_model_feature_columns(ctx, model_manifest),
				ctx.windows,
				ctx.manifest.runtime.seed,
				model_manifest.config,
			)

			model_dir = ctx.artifacts_paths_map.evaluation / str(model_name)

			log_model_runner_end(result, model_dir)

			model_catalog_rows.append(
				{
					"record_type": "model",
					"name": result.info.model_name,
					"status": "completed",
					"wall_time_seconds": result.wall_time_seconds,
				}
			)
			model_feature_usage_rows.append(
				{
					"model_name": result.info.model_name,
					"model_family": result.info.model_family,
					"feature_count": len(result.used_feature_columns),
					"used_feature_columns": " | ".join(result.used_feature_columns),
				}
			)

			metrics_overall, metrics_by_horizon = aggregate_metrics(result.predictions)
			prob_metrics_overall, prob_metrics_by_horizon = aggregate_probabilistic_metrics(
				result.predictions
			)

			ctx.model_artifact_payloads.append(
				ExperimentModelArtifactPayload(
					model_dir=model_dir,
					result=result,
					model_manifest=model_manifest,
					model_artifact=None,
					metrics_overall=metrics_overall,
					metrics_by_horizon=metrics_by_horizon,
					probabilistic_metrics_overall=prob_metrics_overall,
					probabilistic_metrics_by_horizon=prob_metrics_by_horizon,
				)
			)

			if not result.predictions.empty:
				all_predictions.append(result.predictions)

		ctx.predictions = pd.concat(
			all_predictions,
			ignore_index=True) if all_predictions else pd.DataFrame()
		ctx.model_catalog_rows = model_catalog_rows
		ctx.model_feature_usage_rows = model_feature_usage_rows
		ctx.run_catalog_rows.extend(model_catalog_rows)

	def _resolve_model_feature_columns(
		self,
		ctx: ExperimentPipelineContext,
		model_manifest: ModelManifest,
	) -> list[str]:
		return resolve_model_feature_columns(
			model_manifest,
			ctx.feature_columns,
			ctx.dataset.aggregation_level
		)
