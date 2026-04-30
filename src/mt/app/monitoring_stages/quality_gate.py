from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.monitoring.metrics import build_quality_gate_summary


class MonitoringQualityGatePipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		ctx.quality_gate_summary = build_quality_gate_summary(
			metrics=ctx.monitoring_metrics,
			max_recent_wape=ctx.manifest.quality_gate.max_recent_wape,
			max_distribution_shift_score=ctx.manifest.drift.max_distribution_shift_score,
			max_zero_share_delta=ctx.manifest.drift.max_zero_share_delta,
			max_row_count_delta=ctx.manifest.drift.max_row_count_delta,
			max_alert_score=ctx.manifest.quality_gate.max_alert_score,
			issues=ctx.monitoring_issues,
		)
