from dataclasses import dataclass
from pathlib import Path

from mt.domain.artifact.artifact import BaseArtifactPathsMap


@dataclass(slots=True)
class ForecastArtifactPathsMap(BaseArtifactPathsMap):
	dataset: Path
	forecast: Path
	plots: Path

	@classmethod
	def ensure(cls, root: str | Path) -> "ForecastArtifactPathsMap":
		root = Path(root)

		artifact_paths_map = cls(
			**cls._paths(root),
			dataset=root / "dataset",
			forecast=root / "forecast",
			plots=root / "plots",
		)

		cls._ensure_dirs(
			artifact_paths_map.root,
			artifact_paths_map.report,
			artifact_paths_map.run,
			artifact_paths_map.dataset,
			artifact_paths_map.forecast,
			artifact_paths_map.plots,
		)

		return artifact_paths_map

	def dataset_file(self, filename: str) -> Path:
		return self.dataset / filename

	def forecast_file(self, filename: str) -> Path:
		return self.forecast / filename

	def plot_file(self, filename: str) -> Path:
		return self.plots / filename
