from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from mt.domain.artifact.artifact import BaseArtifactPathsMap


@dataclass(slots=True)
class AuditArtifactPathsMap(BaseArtifactPathsMap):
	preparation: Path
	series: Path

	@classmethod
	def ensure(cls, root: str | Path) -> "AuditArtifactPathsMap":
		root = Path(root)

		artifact_paths_map = cls(
			**cls._paths(root),
			preparation=root / "preparation",
			series=root / "series",
		)

		cls._ensure_dirs(
			artifact_paths_map.root,
			artifact_paths_map.report,
			artifact_paths_map.run,
			artifact_paths_map.preparation,
			artifact_paths_map.series,
		)

		return artifact_paths_map

	def preparation_file(self, filename: str) -> Path:
		return self.preparation / filename

	def series_dir(self, series_id: str) -> Path:
		return self.series / _slugify_name(series_id)


@dataclass(slots=True)
class AuditArtifactData:
	summary: pd.DataFrame
	dataset_profile: pd.DataFrame
	category_summary: pd.DataFrame
	sku_summary: pd.DataFrame
	series_diagnostic_table: pd.DataFrame
	data_dictionary: pd.DataFrame
	series_feature_snapshots: dict[str, pd.DataFrame]
	report_lines: list[str]


def _slugify_name(value: str) -> str:
	slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
	return slug.strip("_") or "unknown"
