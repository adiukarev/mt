import pandas as pd

from mt.domain.monitoring.monitoring_pipeline_context import MonitoringPipelineContext
from mt.infra.artifact.text_writer import write_csv, write_markdown, write_yaml


def write_monitoring_artifacts(ctx: MonitoringPipelineContext) -> None:
	if isinstance(ctx.dataset, pd.DataFrame) and not ctx.dataset.empty:
		write_csv(ctx.artifacts_paths_map.data_file("dataset_full.csv"), ctx.dataset)
	if isinstance(ctx.reference_frame, pd.DataFrame) and not ctx.reference_frame.empty:
		write_csv(ctx.artifacts_paths_map.data_file("dataset_reference.csv"), ctx.reference_frame)
	if isinstance(ctx.recent_actuals, pd.DataFrame) and not ctx.recent_actuals.empty:
		write_csv(ctx.artifacts_paths_map.data_file("recent_actuals.csv"), ctx.recent_actuals)
	if isinstance(ctx.predictions, pd.DataFrame) and not ctx.predictions.empty:
		write_csv(ctx.artifacts_paths_map.data_file("predictions.csv"), ctx.predictions)

	write_yaml(ctx.artifacts_paths_map.metrics_file("monitoring_metrics.yaml"), ctx.monitoring_metrics)
	write_yaml(
		ctx.artifacts_paths_map.metrics_file("quality_gate.yaml"),
		ctx.quality_gate_summary,
	)
	write_yaml(
		ctx.artifacts_paths_map.run_file("decision.yaml"),
		ctx.require_decision_artifact().to_dict(),
	 )
	write_yaml(
		ctx.artifacts_paths_map.run_file("source_descriptor.yaml"),
		ctx.source_descriptor,
	)
	if ctx.champion_model is not None:
		write_yaml(
			ctx.artifacts_paths_map.model_file("champion_model_source.yaml"),
			{
				"model_name": ctx.champion_model.model_name.value,
				"source_dir": str(ctx.champion_model.source_dir),
				"source_descriptor": ctx.champion_model.source_descriptor or {},
			},
		)

	write_markdown(ctx.artifacts_paths_map.report_file("REPORT.md"), _build_report_lines(ctx))


def _build_report_lines(ctx: MonitoringPipelineContext) -> list[str]:
	decision = ctx.require_decision()
	return [
		"# Monitoring",
		"",
		"## Summary",
		f"- Decision: `{decision.action.value}`",
		f"- Alert level: `{decision.alert_level}`",
		f"- Should run experiment: `{decision.should_run_experiment}`",
		f"- Champion model: `{ctx.champion_model.model_name.value if ctx.champion_model is not None else 'unavailable'}`",
		"",
		"## Monitoring Metrics",
		*[f"- `{key}`: {value}" for key, value in sorted(ctx.monitoring_metrics.items())],
		"",
		"## Quality Gate",
		f"- Passed: `{ctx.quality_gate_summary.get('passed', False)}`",
		f"- Reasons: {', '.join(ctx.quality_gate_summary.get('reasons', [])) or 'none'}",
	]
