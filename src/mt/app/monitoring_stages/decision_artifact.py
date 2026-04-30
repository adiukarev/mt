from mt.domain.monitoring.monitoring_decision import MonitoringDecision, MonitoringDecisionAction
from mt.domain.monitoring.monitoring_decision_artifact import MonitoringDecisionArtifact
from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage


class MonitoringDecisionArtifactPipelineStage(BasePipelineStage):
	def execute(self, ctx: MonitoringPipelineContext) -> None:
		quality_summary = ctx.quality_gate_summary
		reasons = list(quality_summary.get("reasons", []))
		passed = bool(quality_summary.get("passed", False))
		alert_level = str(quality_summary.get("alert_level", "info"))
		decision_action = str(quality_summary.get("decision_action", MonitoringDecisionAction.NO_ACTION.value))
		action = MonitoringDecisionAction(decision_action)
		should_run_experiment = action == MonitoringDecisionAction.RETRAIN_REQUIRED

		ctx.decision = MonitoringDecision(
			action=action,
			alert_level=alert_level,
			quality_gate_passed=passed,
			should_run_experiment=should_run_experiment,
			should_promote=False,
			reasons=reasons,
			metadata={"alert_score": quality_summary.get("alert_score", 0.0)},
		)
		ctx.decision_artifact = MonitoringDecisionArtifact(
			decision_action=ctx.decision.action.value,
			should_run_experiment=ctx.decision.should_run_experiment,
			quality_gate_passed=ctx.decision.quality_gate_passed,
			alert_level=ctx.decision.alert_level,
			reasons=ctx.decision.reasons,
			monitoring_metrics=ctx.monitoring_metrics,
			source_descriptor=ctx.source_descriptor,
		)
