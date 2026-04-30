from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle, DatasetLoadData
from mt.domain.dataset.dataset_kind import DatasetKind
from mt.infra.dataset.adapters.base import BaseDatasetAdapter
from mt.infra.dataset.weekly_builders.sales_format import build_weekly_sales_format


class M5DatasetAdapter(BaseDatasetAdapter):
	@property
	def kind(self) -> DatasetKind:
		return DatasetKind.M5

	def load(self) -> DatasetLoadData:
		path = Path(self.manifest.path)
		return DatasetLoadData(
			kind=self.kind,
			tables={
				"sales": pd.read_csv(path / "sales_train_evaluation.csv"),
				"calendar": pd.read_csv(path / "calendar.csv", usecols=["d", "date", "wm_yr_wk"]),
			},
			metadata={"source_root": str(path)},
		)

	def prepare(self, data: DatasetLoadData) -> DatasetBundle:
		sales = data.require_table("sales")
		calendar = data.require_table("calendar").copy()

		calendar["date"] = pd.to_datetime(calendar["date"], utc=False)
		calendar["week_start"] = calendar["date"] - pd.to_timedelta(calendar["date"].dt.weekday,
		                                                            unit="D")

		if self.manifest.aggregation_level == "category":
			group_columns = ["cat_id"]
		elif self.manifest.aggregation_level == "sku":
			group_columns = ["item_id", "cat_id"]
		else:
			raise ValueError(f"Unsupported aggregation level for M5: {self.manifest.aggregation_level}")

		weekly = build_weekly_sales_format(
			sales,
			calendar,
			group_columns,
			self.manifest.aggregation_level,
			self.manifest.target_name,
		)

		if self.manifest.series_allowlist is not None:
			weekly = _apply_series_allowlist(weekly, self.manifest.series_allowlist)
		if self.manifest.series_limit is not None:
			weekly = _apply_series_limit(weekly, self.manifest.series_limit)

		raw_context = self._build_raw_context(sales, calendar)
		metadata = {
			"dataset_kind": self.kind,
			"source_frequency": "daily",
			"weekly_rule": self.manifest.week_anchor,
			"stockout_risk": "high",
			"series_limit": self.manifest.series_limit,
			"series_count_after_sampling": int(weekly["series_id"].nunique()),
			"dataset_adapter": self.__class__.__name__,
			"business_category_definition": "Native M5 cat_id hierarchy",
			"raw_input_object": "sales_train_evaluation + calendar",
			"raw_context_label": "raw M5 daily context",
		}
		policy_flags = {
			"target_proxy": False,
			"sku_supported": True,
		}

		return DatasetBundle(
			kind=self.kind,
			adapter_name=self.__class__.__name__,
			aggregation_level=self.manifest.aggregation_level,
			target_name=self.manifest.target_name,
			weekly=weekly,
			metadata=metadata,
			policy_flags=policy_flags,
			raw_context=raw_context,
		)

	def build_raw_context(
		self,
		data: DatasetLoadData,
		bundle: DatasetBundle,
	) -> dict[str, Any]:
		return dict(bundle.raw_context)

	def _build_raw_context(
		self,
		sales: pd.DataFrame,
		calendar: pd.DataFrame,
	) -> dict[str, Any]:
		day_columns = [column for column in sales.columns if column.startswith("d_")]
		sample_day_columns = day_columns[:8]
		daily_total_sales = (
			sales.loc[:, day_columns]
			.sum(axis=0)
			.rename_axis("d")
			.reset_index(name="sales_units")
			.merge(calendar.loc[:, ["d", "date"]], on="d", how="left")
		)
		daily_total_sales["date"] = pd.to_datetime(daily_total_sales["date"], utc=False)
		item_counts_by_category = (
			sales.groupby("cat_id", as_index=False)["item_id"]
			.nunique()
			.rename(columns={"cat_id": "category", "item_id": "item_count"})
		)
		raw_sales_sample = sales.loc[
			:,
			["item_id", "dept_id", "cat_id", "store_id", "state_id", *sample_day_columns],
		].head(10).reset_index(drop=True)
		calendar_sample = calendar.loc[:, ["d", "date", "wm_yr_wk"]].head(10).reset_index(drop=True)

		return {
			"dataset_kind": self.kind,
			"dataset_adapter": self.__class__.__name__,
			"raw_input_object": "sales_train_evaluation + calendar",
			"raw_context_label": "raw M5 daily context",
			"raw_item_count": int(sales["item_id"].nunique()),
			"raw_store_count": int(sales["store_id"].nunique()),
			"raw_state_count": int(sales["state_id"].nunique()),
			"raw_department_count": int(sales["dept_id"].nunique()),
			"raw_daily_observations": len(day_columns),
			"daily_total_sales": daily_total_sales.loc[:, ["date", "sales_units"]].sort_values(
				"date").reset_index(drop=True),
			"item_counts_by_category": item_counts_by_category,
			"raw_sales_sample": raw_sales_sample,
			"calendar_sample": calendar_sample,
			"calendar_period_start": str(pd.to_datetime(calendar["date"]).min().date()),
			"calendar_period_end": str(pd.to_datetime(calendar["date"]).max().date()),
		}


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
		raise ValueError("series_allowlist did not match any M5 series_id")
	return filtered.sort_values(["series_id", "week_start"]).reset_index(drop=True)
