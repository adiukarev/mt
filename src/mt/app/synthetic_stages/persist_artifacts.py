from __future__ import annotations

from time import perf_counter

import yaml

from mt.domain.stage import BaseStage
from mt.domain.synthetic_generation import SyntheticGenerationPipelineContext
from mt.infra.artifact.writer import write_csv, write_markdown
from mt.infra.synthetic.generator import build_generation_report, build_preview_artifacts


class SyntheticPersistArtifactsStage(BaseStage):
	name = "synthetic_persist_artifacts"

	def execute(self, ctx: SyntheticGenerationPipelineContext) -> None:
		if ctx.dataset is None or ctx.metadata is None:
			raise ValueError()

		write_csv(
			ctx.artifacts_paths_map.dataset / f"{ctx.manifest.runtime.dataset_name}.csv",
			ctx.dataset,
		)
		write_csv(ctx.artifacts_paths_map.dataset / "series_metadata.csv", ctx.metadata)
		write_markdown(
			ctx.artifacts_paths_map.dataset / "README.md",
			build_generation_report(
				ctx.manifest,
				ctx.dataset,
				ctx.metadata,
			).splitlines(),
		)

		build_preview_artifacts(ctx.dataset, ctx.artifacts_paths_map.preview)

		(ctx.artifacts_paths_map.run / "generation_manifest_snapshot.yaml").write_text(
			yaml.safe_dump(ctx.manifest.as_dict(), sort_keys=False, allow_unicode=True),
			encoding="utf-8",
		)
