from mt.domain.synthetic_generation.synthetic_generation_pipeline_context import SyntheticGenerationPipelineContext
from mt.infra.artifact.text_writer import write_markdown


def write_synthetic_report(ctx: SyntheticGenerationPipelineContext) -> None:
	ctx.require_dataset()
	ctx.require_metadata()

	lines = [
		"# Synthetic Weekly Dataset",
		"",
		"## Summary",
		f"- Data root: `{ctx.dataset_root}`",
		f"- Series: `{ctx.generation_summary.get('series_count', 0)}`",
		f"- Scenarios: `{ctx.generation_summary.get('scenario_count', 0)}`",
		f"- Rows: `{ctx.generation_summary.get('row_count', 0)}`",
		f"- History rows: `{ctx.generation_summary.get('history_rows', 0)}`",
		f"- Recent rows: `{ctx.generation_summary.get('recent_rows', 0)}`",
		f"- Batches: `{ctx.generation_summary.get('batch_count', 0)}`",
		"",
		"## Notes",
		"- Канонический synthetic dataset materialize-ится в `data/...`, а не хранится отдельной копией в `artifacts/...`.",
		"- Artifact tree intentionally содержит только manifest/summary/report metadata для smoke- и orchestration-проверок.",
	]
	write_markdown(ctx.artifacts_paths_map.report_file("REPORT.md"), lines)
