from mt.domain.audit.audit_artifact import AuditArtifactData
from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.infra.artifact.text_writer import write_markdown


def write_audit_report(ctx: AuditPipelineContext, audit_artifacts: AuditArtifactData) -> None:
	write_markdown(ctx.artifacts_paths_map.report_file("REPORT.md"), audit_artifacts.report_lines)
