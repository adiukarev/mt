from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.model.model_name import ModelName

from mt.domain.artifact.artifact import BaseArtifactPathsMap
from mt.domain.probabilistic.probabilistic import ProbabilisticSource


@dataclass(slots=True)
class ModelArtifactData:
	model_name: ModelName
	dataset_manifest: DatasetManifest
	feature_manifest: FeatureManifest
	feature_columns: list[str]
	horizons: list[int]
	adapters_by_horizon: dict[int, Any]
	training_aggregation_level: str
	training_last_week_start: str
	probabilistic_quantiles: list[float]
	interval_levels: list[float]
	probabilistic_source_by_horizon: dict[int, ProbabilisticSource]
	model_config: dict[str, Any] | None = None
	conformal_calibrator_state: dict[str, Any] | None = None
	probabilistic_metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelArtifactPathsMap(BaseArtifactPathsMap):
	model: Path

	@classmethod
	def ensure(cls, root: str | Path) -> "ModelArtifactPathsMap":
		root = Path(root)

		artifact_paths_map = cls(
			**cls._paths(root),
			model=root / "model",
		)

		cls._ensure_dirs(
			artifact_paths_map.root,
			artifact_paths_map.report,
			artifact_paths_map.run,
			artifact_paths_map.model,
		)

		return artifact_paths_map

	def model_file(self, filename: str) -> Path:
		return self.model / filename
