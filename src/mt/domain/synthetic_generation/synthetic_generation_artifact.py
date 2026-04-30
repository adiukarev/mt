from dataclasses import dataclass
from pathlib import Path

from mt.domain.artifact.artifact import BaseArtifactPathsMap


@dataclass(slots=True)
class SyntheticGenerationArtifactPathsMap(BaseArtifactPathsMap):
	@classmethod
	def ensure(cls, root: str | Path) -> "SyntheticGenerationArtifactPathsMap":
		root = Path(root)

		artifact_paths_map = cls(
			**cls._paths(root),
		)

		cls._ensure_dirs(
			artifact_paths_map.root,
			artifact_paths_map.report,
			artifact_paths_map.run,
		)

		return artifact_paths_map
