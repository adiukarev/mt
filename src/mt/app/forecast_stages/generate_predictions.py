from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.forecast.frame import infer_horizon
from mt.infra.forecast.inference_policy import resolve_forecast_inference_policy
from mt.infra.forecast.metrics_builder import build_metrics, build_probabilistic_metrics
from mt.infra.forecast.prediction_builder import build_saved_model_predictions


class ForecastGeneratePredictionsPipelineStage(BasePipelineStage):
	def execute(self, ctx: ForecastPipelineContext) -> None:
		frame = ctx.require_frame()
		reference_model = ctx.require_reference_model()
		dataset = ctx.require_dataset()

		ctx.resolved_horizon = ctx.manifest.forecast.horizon_weeks or infer_horizon(frame)

		policy = resolve_forecast_inference_policy(
			dataset_bundle=dataset,
			reference_model=reference_model,
			horizon_weeks=ctx.resolved_horizon,
		)
		ctx.runtime_metadata["forecast_inference_mode"] = policy["mode"]
		ctx.runtime_metadata["forecast_inference_reasons"] = policy["reasons"]

		ctx.predictions = build_saved_model_predictions(
			frame=frame,
			reference_model=reference_model,
			horizon_weeks=ctx.resolved_horizon,
		)

		ctx.metrics = build_metrics(ctx.predictions)
		ctx.probabilistic_metrics = build_probabilistic_metrics(ctx.predictions)
