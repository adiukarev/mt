from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle, DatasetLoadData
from mt.domain.dataset.dataset_kind import DatasetKind
from mt.infra.dataset.adapters.base import BaseDatasetAdapter


class SyntheticDatasetAdapter(BaseDatasetAdapter):
	kind: DatasetKind = DatasetKind.SYNTHETIC

	def load(self) -> DatasetLoadData:
		path = Path(self.manifest.path)
		dataset_path = path if path.is_file() else path / "dataset.csv"
		frame = pd.read_csv(dataset_path, parse_dates=["week_start"])

		return DatasetLoadData(
			kind=self.kind,
			tables={"weekly": frame},
			metadata={"source_root": str(path), "dataset_path": str(dataset_path)},
		)

	def prepare(self, data: DatasetLoadData) -> DatasetBundle:
		weekly = data.require_table("weekly").copy()
		required_columns = {"series_id", "category", "week_start", "sales_units"}
		missing = required_columns.difference(weekly.columns)
		if missing:
			raise ValueError(f"Synthetic dataset is missing columns: {sorted(missing)}")

		weekly["week_start"] = pd.to_datetime(weekly["week_start"], utc=False)

		if self.manifest.aggregation_level == "category":
			group_columns = ["category", "week_start"]
			if "scenario_name" in weekly.columns:
				group_columns = ["scenario_name", *group_columns]
			weekly = (
				weekly.groupby(group_columns, as_index=False)
				.agg(sales_units=("sales_units", "sum"))
				.assign(series_id=lambda df: df["category"])
			)

		elif self.manifest.aggregation_level != "sku":
			raise ValueError()

		if self.manifest.series_allowlist is not None:
			weekly = _apply_series_allowlist(weekly, self.manifest.series_allowlist)
		if self.manifest.series_limit is not None:
			weekly = _apply_series_limit(weekly, self.manifest.series_limit)

		metadata = {
			"dataset_kind": self.kind,
			"source_frequency": "weekly",
			"weekly_rule": self.manifest.week_anchor,
			"stockout_risk": "synthetic_not_modeled",
			"dataset_adapter": self.__class__.__name__,
			"business_category_definition": "Synthetic category provided by generator",
			"raw_input_object": "synthetic weekly dataset.csv",
			"raw_context_label": "raw synthetic weekly context",
		}
		policy_flags = {
			"target_proxy": False,
			"sku_supported": True,
		}
		raw_context = {
			"dataset_kind": self.kind,
			"dataset_adapter": self.__class__.__name__,
			"raw_input_object": "synthetic weekly dataset.csv",
			"raw_context_label": "raw synthetic weekly context",
			"daily_total_sales": pd.DataFrame(),
			"item_counts_by_category": pd.DataFrame(),
			"raw_sales_sample": weekly.head(10).reset_index(drop=True),
			"calendar_sample": weekly.loc[:, ["week_start"]].head(10).reset_index(drop=True),
			"calendar_period_start": str(weekly["week_start"].min().date()),
			"calendar_period_end": str(weekly["week_start"].max().date()),
		}

		return DatasetBundle(
			kind=self.kind,
			adapter_name=self.__class__.__name__,
			aggregation_level=self.manifest.aggregation_level,
			target_name=self.manifest.target_name,
			weekly=weekly.sort_values(["series_id", "week_start"]).reset_index(drop=True),
			metadata=metadata,
			policy_flags=policy_flags,
			raw_context=raw_context,
		)


def _apply_series_limit(weekly: pd.DataFrame, series_limit: int) -> pd.DataFrame:
	series_totals = (
		weekly.groupby("series_id", as_index=False)
		.agg(total_sales_units=("sales_units", "sum"))
		.sort_values(["total_sales_units", "series_id"], ascending=[False, True])
	)
	selected_series = series_totals.head(series_limit)["series_id"]
	return (
		weekly[weekly["series_id"].isin(selected_series)]
		.sort_values(["series_id", "week_start"])
		.reset_index(drop=True)
	)


def _apply_series_allowlist(weekly: pd.DataFrame, series_allowlist: list[str]) -> pd.DataFrame:
	filtered = weekly[weekly["series_id"].astype(str).isin(series_allowlist)].copy()
	if filtered.empty:
		raise ValueError("series_allowlist did not match any synthetic series_id")
	return filtered.sort_values(["series_id", "week_start"]).reset_index(drop=True)
