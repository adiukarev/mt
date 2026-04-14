from __future__ import annotations

from pathlib import Path
import pandas as pd

from mt.app.predict_stages.dataset_preparation import PredictDatasetPreparationStage
from mt.app.predict_stages.persist_artifacts import PredictPersistArtifactsStage
from mt.app.predict_stages.reference_model_resolution import PredictReferenceModelResolutionStage
from mt.app.predict_stages.forecast_generation import PredictForecastGenerationStage
from mt.domain.pipeline import BasePipeline
from mt.domain.predict import PredictArtifactPathsMap, PredictPipelineContext
from mt.domain.predict_manifest import SyntheticPredictManifest
from mt.infra.artifact.versioning import archive_existing_artifacts
from mt.infra.artifact.writer import write_csv


class PredictPipeline(BasePipeline):
	def __init__(self) -> None:
		super().__init__([
			PredictReferenceModelResolutionStage(),
			PredictDatasetPreparationStage(),
			PredictForecastGenerationStage(),
			PredictPersistArtifactsStage(),
		])

	def build_context(self, manifest: SyntheticPredictManifest) -> PredictPipelineContext:
		output_dir = manifest.runtime.output_dir

		archive_existing_artifacts(output_dir)

		return PredictPipelineContext(
			manifest=manifest,
			artifacts_paths_map=self._ensure_artifacts_paths_map(output_dir),
		)

	def finalize(self, ctx: PredictPipelineContext) -> PredictPipelineContext:
		# artifacts
		self._persist_artifacts(ctx)

		return ctx

	def _persist_artifacts(self, ctx: PredictPipelineContext) -> None:
		write_csv(
			ctx.artifacts_paths_map.run / "run_catalog.csv",
			pd.DataFrame([
				{
					"record_type": "pipeline",
					"name": "predict",
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

	def _ensure_artifacts_paths_map(self, output_dir: str) -> PredictArtifactPathsMap:
		root_path = Path(output_dir)
		dataset_dir = root_path / "dataset"
		forecast_dir = root_path / "forecast"
		run_dir = root_path / "run"

		for path in (root_path, dataset_dir, forecast_dir, run_dir):
			path.mkdir(parents=True, exist_ok=True)

		return PredictArtifactPathsMap(
			root=root_path,
			dataset=dataset_dir,
			forecast=forecast_dir,
			run=run_dir,
		)
