import shutil
from pathlib import Path

from mt.domain.synthetic_generation.synthetic_generation_pipeline_context import SyntheticGenerationPipelineContext
from mt.infra.synthetic_generation_artifact.report.text import write_synthetic_report
from mt.infra.synthetic_generation_artifact.run.text import (
	write_generation_run_summary,
	write_manifest_snapshot,
)


def write_synthetic_generation_artifacts(ctx: SyntheticGenerationPipelineContext) -> None:
	_cleanup_legacy_artifacts(ctx)

	# report
	write_synthetic_report(ctx)

	# run
	write_manifest_snapshot(ctx)
	write_generation_run_summary(ctx)


def _cleanup_legacy_artifacts(ctx: SyntheticGenerationPipelineContext) -> None:
	for dirname in ("dataset", "preview", "plots"):
		shutil.rmtree(Path(ctx.artifacts_paths_map.root) / dirname, ignore_errors=True)
	try:
		(ctx.artifacts_paths_map.report_file("synthetic_report.md")).unlink()
	except FileNotFoundError:
		pass
