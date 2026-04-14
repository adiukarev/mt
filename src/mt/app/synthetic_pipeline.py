from __future__ import annotations

from pathlib import Path

import pandas as pd

from mt.app.synthetic_stages.dataset_generation import SyntheticDatasetGenerationStage
from mt.app.synthetic_stages.demo_forecast import SyntheticDemoForecastStage
from mt.app.synthetic_stages.persist_artifacts import SyntheticPersistArtifactsStage
from mt.domain.pipeline import BasePipeline
from mt.domain.synthetic import SyntheticManifest
from mt.domain.synthetic_generation import (
	SyntheticArtifactPathsMap,
	SyntheticGenerationPipelineContext,
)
from mt.infra.artifact.versioning import archive_existing_artifacts
from mt.infra.artifact.writer import write_csv, write_markdown


class SyntheticGenerationPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__([
			SyntheticDatasetGenerationStage(),
			SyntheticDemoForecastStage(),
			SyntheticPersistArtifactsStage(),
		])

	def build_context(self, manifest: SyntheticManifest) -> SyntheticGenerationPipelineContext:
		output_dir = manifest.runtime.output_dir

		archive_existing_artifacts(output_dir)

		return SyntheticGenerationPipelineContext(
			manifest=manifest,
			artifacts_paths_map=self._ensure_artifacts_paths_map(output_dir),
		)

	def finalize(self, ctx: SyntheticGenerationPipelineContext) -> SyntheticGenerationPipelineContext:
		# artifacts
		self._persist_artifacts(ctx)

		return ctx

	def _persist_artifacts(self, ctx: SyntheticGenerationPipelineContext) -> None:
		write_csv(
			ctx.artifacts_paths_map.run / "run_catalog.csv",
			pd.DataFrame([
				{
					"record_type": "pipeline",
					"name": "synthetic",
					"status": "completed",
					"wall_time_seconds": ctx.pipeline_wall_time_seconds,
				},
				*[
					{
						"record_type": "stage",
						"name": str(stage_info["stage_name"]),
						"status": str(stage_info["status"]),
						"wall_time_seconds": stage_info["wall_time_seconds"],
					}
					for stage_info in ctx.stage_timings
				],
			])
		)

	def _ensure_artifacts_paths_map(self, output_dir: str) -> SyntheticArtifactPathsMap:
		root = Path(output_dir)
		dataset_dir = root / "dataset"
		preview_dir = root / "preview"
		run_dir = root / "run"

		for path in (root, dataset_dir, preview_dir, run_dir):
			path.mkdir(parents=True, exist_ok=True)

		return SyntheticArtifactPathsMap(
			root=root,
			dataset=dataset_dir,
			preview=preview_dir,
			run=run_dir,
		)
