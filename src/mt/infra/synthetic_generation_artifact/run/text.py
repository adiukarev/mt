from pathlib import Path

from mt.domain.synthetic_generation.synthetic_generation_pipeline_context import SyntheticGenerationPipelineContext
from mt.infra.artifact.text_writer import write_yaml
from mt.infra.observability.runtime.summary_builder import build_tracking_summary


def write_manifest_snapshot(ctx: SyntheticGenerationPipelineContext) -> None:
	write_yaml(
		ctx.artifacts_paths_map.run_file("manifest_snapshot.yaml"),
		ctx.manifest.to_dict(),
	)


def write_generation_run_summary(ctx: SyntheticGenerationPipelineContext) -> None:
	ctx.require_dataset()

	payload = {
		"dataset_root": ctx.dataset_root,
		"materialized_paths": ctx.materialized_paths,
		**ctx.generation_summary,
		**build_tracking_summary(ctx),
	}
	write_yaml(ctx.artifacts_paths_map.run_file("summary.yaml"), payload)
