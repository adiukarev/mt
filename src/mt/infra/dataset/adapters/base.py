from abc import ABC
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset.dataset import DatasetAdapter, DatasetBundle, DatasetLoadData


class BaseDatasetAdapter(DatasetAdapter, ABC):
	def build_raw_context(
		self,
		data: DatasetLoadData,
		bundle: DatasetBundle,
	) -> dict[str, Any]:
		return dict(bundle.raw_context)


def read_canonical_forecast_frame(
	dataset_path: str | Path,
	aggregation_level: str | None = None,
) -> pd.DataFrame:
	frame = pd.read_csv(dataset_path, parse_dates=["week_start"])
	required_columns = {"series_id", "category", "week_start", "is_history", "sales_units"}

	missing = required_columns.difference(frame.columns)
	if missing:
		raise ValueError()

	if aggregation_level is not None:
		frame = aggregate_canonical_forecast_frame(frame, aggregation_level)

	return frame.sort_values(["series_id", "week_start"]).reset_index(drop=True)


def aggregate_canonical_forecast_frame(frame: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
	if aggregation_level == "sku":
		return frame.copy()
	if aggregation_level != "category":
		raise ValueError(f"Unsupported aggregation level: {aggregation_level}")

	group_columns = ["category", "week_start", "is_history"]

	aggregations: dict[str, str] = {"sales_units": "sum"}
	for column, reducer in (
			("stockout_flag", "max"),
			("demand_noise_scale", "mean"),
			("target_name", "first"),
	):
		if column in frame.columns:
			aggregations[column] = reducer

	aggregated = frame.groupby(group_columns, as_index=False).agg(aggregations)
	aggregated["series_id"] = aggregated["category"]
	return aggregated
