from dataclasses import dataclass
from pathlib import Path

from mt.domain.artifact.artifact import BaseArtifactPathsMap


@dataclass(slots=True)
class MonitoringArtifactPathsMap(BaseArtifactPathsMap):
	data: Path
	metrics: Path
	models: Path

	@classmethod
	def ensure(cls, root: str | Path) -> "MonitoringArtifactPathsMap":
		root = Path(root)
		artifact_paths_map = cls(
			**cls._paths(root),
			data=root / "data",
			metrics=root / "metrics",
			models=root / "models",
		)
		cls._ensure_dirs(
			artifact_paths_map.root,
			artifact_paths_map.report,
			artifact_paths_map.run,
			artifact_paths_map.data,
			artifact_paths_map.metrics,
			artifact_paths_map.models,
		)
		return artifact_paths_map

	def data_file(self, filename: str) -> Path:
		return self.data / filename

	def metrics_file(self, filename: str) -> Path:
		return self.metrics / filename

	def model_file(self, filename: str) -> Path:
		return self.models / filename
