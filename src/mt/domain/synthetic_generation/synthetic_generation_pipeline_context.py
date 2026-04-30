from dataclasses import dataclass, field

import pandas as pd

from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.domain.synthetic_generation.synthetic_generation_artifact import \
	SyntheticGenerationArtifactPathsMap
from mt.domain.synthetic_generation.synthetic_generation_pipeline_manifest import \
	SyntheticGenerationPipelineManifest


@dataclass(slots=True)
class SyntheticGenerationPipelineContext(BasePipelineContext):
	manifest: SyntheticGenerationPipelineManifest
	artifacts_paths_map: SyntheticGenerationArtifactPathsMap | None = None

	dataset: pd.DataFrame | None = None
	metadata: pd.DataFrame | None = None
	dataset_root: str | None = None
	materialized_paths: list[str] = field(default_factory=list)
	generation_summary: dict[str, object] = field(default_factory=dict)

	def __post_init__(self) -> None:
		self.artifacts_paths_map = SyntheticGenerationArtifactPathsMap.ensure(
			self.manifest.runtime.artifacts_dir)

	def require_dataset(self) -> pd.DataFrame:
		return self.require_value(self.dataset, "dataset")

	def require_metadata(self) -> pd.DataFrame:
		return self.require_value(self.metadata, "metadata")
