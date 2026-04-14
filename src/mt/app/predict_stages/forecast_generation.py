from __future__ import annotations

from time import perf_counter

from mt.domain.predict import PredictPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.synthetic.predict import build_metrics, build_saved_model_predictions, \
	infer_horizon


class PredictForecastGenerationStage(BaseStage):
	name = "predict_forecast_generation"

	def execute(self, ctx: PredictPipelineContext) -> None:
		if ctx.frame is None or ctx.reference_model is None:
			raise ValueError()

		ctx.resolved_horizon = ctx.manifest.forecast.horizon_weeks or infer_horizon(ctx.frame)
		ctx.predictions = build_saved_model_predictions(
			frame=ctx.frame,
			reference_model=ctx.reference_model,
			horizon_weeks=ctx.resolved_horizon,
		)
		ctx.metrics = build_metrics(ctx.predictions)
