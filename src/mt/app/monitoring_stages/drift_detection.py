from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.monitoring.metrics import build_monitoring_metrics


class MonitoringDriftDetectionPipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		ctx.monitoring_metrics = build_monitoring_metrics(
			reference_frame=ctx.require_reference_frame(),
			recent_actuals=ctx.require_recent_actuals(),
			predictions=ctx.require_predictions(),
			reference_weeks=ctx.manifest.drift.reference_weeks,
		)
