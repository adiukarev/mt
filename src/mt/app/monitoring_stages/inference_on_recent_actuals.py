import pandas as pd

from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.forecast.prediction_builder import run_saved_model_forecast_window
from mt.infra.observability.logger.runtime_logger import log_warning
from mt.infra.probabilistic.schema import build_empty_prediction_frame


class MonitoringInferenceOnRecentActualsPipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		recent_actuals = ctx.require_recent_actuals()
		if recent_actuals.empty:
			ctx.monitoring_issues.append("recent_actuals_missing")
			ctx.predictions = build_empty_prediction_frame()
			return

		if ctx.champion_model is None or ctx.champion_model.artifact is None:
			ctx.predictions = build_empty_prediction_frame()
			return

		horizon_weeks = int(recent_actuals["week_start"].nunique())
		ctx.source_descriptor["monitoring_horizon_weeks"] = horizon_weeks
		try:
			ctx.predictions = run_saved_model_forecast_window(
				forecast_frame=ctx.require_dataset(),
				horizon_weeks=horizon_weeks,
				artifact=ctx.champion_model.artifact,
			)
			if ctx.predictions.empty:
				ctx.monitoring_issues.append("recent_predictions_unavailable")
		except Exception as error:
			ctx.monitoring_issues.append("recent_predictions_unavailable")
			ctx.predictions = build_empty_prediction_frame()
			ctx.source_descriptor["prediction_error"] = str(error)
			log_warning(
				"monitoring",
				scope="monitoring",
				reason="recent_predictions_unavailable",
				error=str(error),
			)
