from pathlib import Path

from mt.domain.audit.audit_artifact import AuditArtifactData
from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.infra.artifact.text_writer import write_yaml
from mt.infra.observability.runtime.summary_builder import build_tracking_summary


def write_audit_run_artifacts(
	ctx: AuditPipelineContext,
	audit_artifacts: AuditArtifactData,
) -> None:
	if ctx.dataset is None:
		raise ValueError()

	payload = {
		"dataset_kind": ctx.dataset.kind,
		"aggregation_level": ctx.dataset.aggregation_level,
		"series_count": int(ctx.dataset.weekly["series_id"].nunique()),
		"row_count": int(len(ctx.dataset.weekly)),
		"segment_count": int(ctx.segments["segment_label"].nunique()) if ctx.segments is not None else 0,
		**build_tracking_summary(ctx),
	}

	write_yaml(ctx.artifacts_paths_map.run_file("summary.yaml"), payload)
