from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.forecast.frame import build_forecast_frame


class ForecastFrameBuildingPipelineStage(BasePipelineStage):
	def execute(self, ctx: ForecastPipelineContext) -> None:
		dataset = ctx.require_dataset()
		horizon_weeks = ctx.manifest.forecast.horizon_weeks
		if horizon_weeks is None:
			reference_model = ctx.require_reference_model()
			if reference_model.artifact is None:
				raise ValueError()
			horizon_weeks = max(reference_model.artifact.horizons)

		dataset_manifest = ctx.require_resolved_dataset_manifest()
		if dataset_manifest.aggregation_level != dataset.aggregation_level:
			raise ValueError()

		ctx.frame = build_forecast_frame(
			dataset=dataset,
			horizon_weeks=horizon_weeks,
		)
