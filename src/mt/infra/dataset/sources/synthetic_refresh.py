from pathlib import Path

import pandas as pd

from mt.domain.dataset.dataset import DatasetLoadData
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.infra.dataset.sources.base import DatasetSourceRefreshResult, DatasetSourceService

SYNTHETIC_SOURCE_REQUIRED_COLUMNS = frozenset(
	{
		"scenario_name",
		"series_id",
		"category",
		"week_start",
		"sales_units",
	}
)


class SyntheticRefreshDatasetSourceService(DatasetSourceService):
	def refresh_dataset(
		self,
		dataset_manifest: DatasetManifest,
	) -> DatasetSourceRefreshResult:
		root = Path(dataset_manifest.path)
		config = self.source_manifest.source_config
		_require_synthetic_source_tree(root)
		base_frame = _read_synthetic_source_frame(root / "dataset.csv")
		batches_dir = root / "batches"
		batch_files = sorted(batches_dir.glob("*.csv"))
		applied_batch_count = _resolve_applied_batch_count(config, len(batch_files))
		selected_batch_files = batch_files[-applied_batch_count:] if applied_batch_count else []
		recent_actuals = _load_recent_actuals(selected_batch_files, base_frame.columns.tolist())
		base_frame, recent_actuals = _apply_dataset_scope(
			base_frame=base_frame,
			recent_actuals=recent_actuals,
			dataset_manifest=dataset_manifest,
		)
		reference_frame = base_frame.copy()
		reference_frame["is_history"] = True
		if not recent_actuals.empty:
			recent_actuals["is_history"] = False
		if recent_actuals.empty:
			full_frame = reference_frame.copy()
		else:
			full_frame = pd.concat([reference_frame, recent_actuals], ignore_index=True)
		full_frame = full_frame.sort_values(["series_id", "week_start"]).reset_index(drop=True)

		materialized_paths = [str(root / "dataset.csv"), *[str(path) for path in selected_batch_files]]

		return DatasetSourceRefreshResult(
			reference_frame=reference_frame.reset_index(drop=True),
			recent_actuals=recent_actuals.reset_index(drop=True),
			full_frame=full_frame,
			source_descriptor={
				"source_service": self.name,
				"dataset_root": str(root),
				"batches_dir": str(batches_dir),
				"applied_batch_count": applied_batch_count,
				"available_batch_count": len(batch_files),
				"dataset_path": str(root),
			},
			materialized_paths=list(dict.fromkeys(materialized_paths)),
		)


def _require_synthetic_source_tree(root: Path) -> None:
	if not (root / "dataset.csv").exists():
		raise FileNotFoundError(
			f"Synthetic dataset history is missing: {root / 'dataset.csv'}. "
			"Run `mt synthetic-generation --manifest ...` or the Airflow synthetic_generation DAG first."
		)
	if not (root / "batches").is_dir() or not any((root / "batches").glob("*.csv")):
		raise FileNotFoundError(
			f"Synthetic dataset batches are missing under {root / 'batches'}. "
			"Run `mt synthetic-generation --manifest ...` or the Airflow synthetic_generation DAG first."
		)


def _resolve_applied_batch_count(config: dict[str, object], available_batch_count: int) -> int:
	value = config.get("applied_batch_count", available_batch_count)
	applied_batch_count = int(value)
	if applied_batch_count < 0:
		raise ValueError("applied_batch_count must be >= 0")
	return min(applied_batch_count, available_batch_count)


def _load_recent_actuals(batch_files: list[Path], columns: list[str]) -> pd.DataFrame:
	if not batch_files:
		return pd.DataFrame(columns=columns)
	frames = [_read_synthetic_source_frame(path) for path in batch_files]
	return pd.concat(frames, ignore_index=True)


def _read_synthetic_source_frame(path: Path) -> pd.DataFrame:
	frame = pd.read_csv(path, parse_dates=["week_start"])
	_validate_synthetic_source_frame(frame, path)
	frame["week_start"] = pd.to_datetime(frame["week_start"], utc=False)
	frame["series_id"] = frame["series_id"].astype(str)
	frame["category"] = frame["category"].astype(str)
	frame["scenario_name"] = frame["scenario_name"].astype(str)
	frame["sales_units"] = pd.to_numeric(frame["sales_units"], errors="raise")
	return frame.sort_values(["series_id", "week_start"]).reset_index(drop=True)


def _validate_synthetic_source_frame(frame: pd.DataFrame, path: Path) -> None:
	missing = SYNTHETIC_SOURCE_REQUIRED_COLUMNS.difference(frame.columns)
	if missing:
		raise ValueError(f"Synthetic source file {path} is missing columns: {sorted(missing)}")
	if frame.empty:
		raise ValueError(f"Synthetic source file {path} is empty")
	if frame[["series_id", "category", "week_start", "sales_units"]].isna().any().any():
		raise ValueError(f"Synthetic source file {path} has null key or target values")


def _apply_dataset_scope(
	base_frame: pd.DataFrame,
	recent_actuals: pd.DataFrame,
	dataset_manifest: DatasetManifest,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	selected_series = _resolve_selected_series(base_frame, dataset_manifest)
	if selected_series is None:
		return base_frame, recent_actuals

	return (
		_filter_series(base_frame, selected_series),
		_filter_series(recent_actuals, selected_series),
	)


def _resolve_selected_series(
	base_frame: pd.DataFrame,
	dataset_manifest: DatasetManifest,
) -> set[str] | None:
	if dataset_manifest.series_allowlist is not None:
		return set(dataset_manifest.series_allowlist)
	if dataset_manifest.series_limit is None:
		return None

	series_totals = (
		base_frame.groupby("series_id", as_index=False)
		.agg(total_sales_units=("sales_units", "sum"))
		.sort_values(["total_sales_units", "series_id"], ascending=[False, True])
	)
	return set(series_totals.head(dataset_manifest.series_limit)["series_id"].astype(str))


def _filter_series(frame: pd.DataFrame, selected_series: set[str]) -> pd.DataFrame:
	if frame.empty:
		return frame.copy()
	filtered = frame.loc[frame["series_id"].astype(str).isin(selected_series)].copy()
	if filtered.empty:
		raise ValueError("Synthetic source series scope did not match any series_id")
	return filtered.sort_values(["series_id", "week_start"]).reset_index(drop=True)


def build_synthetic_dataset_load_data(
	dataset_manifest: DatasetManifest,
	result: DatasetSourceRefreshResult,
) -> DatasetLoadData:
	return DatasetLoadData(
		kind=dataset_manifest.kind,
		tables={"weekly": result.full_frame.drop(columns=["is_history"], errors="ignore")},
		metadata={
			"source_root": str(dataset_manifest.path),
			"dataset_path": str(Path(dataset_manifest.path) / "dataset.csv"),
			"applied_batch_count": result.source_descriptor.get("applied_batch_count", 0),
			"source_service": result.source_descriptor.get("source_service"),
		},
	)
