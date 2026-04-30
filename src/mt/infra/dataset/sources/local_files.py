from pathlib import Path

import pandas as pd

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.infra.dataset.sources.base import DatasetSourceRefreshResult, DatasetSourceService


class LocalFilesDatasetSourceService(DatasetSourceService):
	def refresh_dataset(
		self,
		dataset_manifest: DatasetManifest,
	) -> DatasetSourceRefreshResult:
		root = Path(dataset_manifest.path)
		dataset_path = root if root.is_file() else root / "dataset.csv"
		frame = pd.read_csv(dataset_path, parse_dates=["week_start"])
		if "is_history" not in frame.columns:
			frame["is_history"] = True
		reference_frame = frame.loc[frame["is_history"].astype(bool)].copy()
		recent_actuals = frame.loc[~frame["is_history"].astype(bool)].copy()
		return DatasetSourceRefreshResult(
			reference_frame=reference_frame.reset_index(drop=True),
			recent_actuals=recent_actuals.reset_index(drop=True),
			full_frame=frame.sort_values(["series_id", "week_start"]).reset_index(drop=True),
			source_descriptor={
				"source_service": self.name,
				"dataset_path": str(dataset_path),
				"applied_batch_count": int(recent_actuals["week_start"].nunique())
				if not recent_actuals.empty else 0,
			},
			materialized_paths=[str(dataset_path)],
		)
