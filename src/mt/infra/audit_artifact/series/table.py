from mt.domain.audit.audit_artifact import AuditArtifactData
from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.infra.artifact.text_writer import write_csv


def write_series_artifacts(
	ctx: AuditPipelineContext,
	audit_artifacts: AuditArtifactData,
) -> None:
	if ctx.dataset is None:
		raise ValueError()

	if audit_artifacts.series_diagnostic_table.empty:
		return

	series_metrics_by_id = {
		str(row["series_id"]): row.to_frame().T.reset_index(drop=True)
		for _, row in audit_artifacts.series_diagnostic_table.iterrows()
	}
	for series_id, snapshot in audit_artifacts.series_feature_snapshots.items():
		artifacts_dir = ctx.artifacts_paths_map.series_dir(series_id)
		metrics = series_metrics_by_id.get(str(series_id))
		if metrics is not None:
			write_csv(artifacts_dir / "series_metrics.csv", metrics)
		write_csv(artifacts_dir / "feature_snapshot.csv", snapshot)
