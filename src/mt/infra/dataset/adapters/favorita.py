from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset.dataset import DatasetBundle, DatasetLoadData
from mt.domain.dataset.dataset_kind import DatasetKind
from mt.infra.dataset.adapters.base import BaseDatasetAdapter


_TRAIN_COLUMNS = ["date", "store_nbr", "item_nbr", "unit_sales"]
_TRAIN_DTYPES = {
	"store_nbr": "int16",
	"item_nbr": "int32",
	"unit_sales": "float32",
}
_TRAIN_CHUNK_SIZE = 2_000_000


class FavoritaDatasetAdapter(BaseDatasetAdapter):
	@property
	def kind(self) -> DatasetKind:
		return DatasetKind.FAVORITA

	def load(self) -> DatasetLoadData:
		path = Path(self.manifest.path)
		return DatasetLoadData(
			kind=self.kind,
			tables={
				"items": pd.read_csv(path / "items.csv"),
				"stores": pd.read_csv(path / "stores.csv"),
				"test": pd.DataFrame(),
			},
			metadata={"source_root": str(path), "train_path": str(path / "train.csv")},
		)

	def prepare(self, data: DatasetLoadData) -> DatasetBundle:
		items = data.require_table("items").copy()
		stores = data.require_table("stores").copy()

		observed_daily, raw_stats = _build_observed_daily_frame(
			data,
			items,
			self.manifest.aggregation_level,
		)
		scoped_observed_daily = _apply_series_scope(
			observed_daily,
			series_allowlist=self.manifest.series_allowlist,
			series_limit=self.manifest.series_limit,
		)
		daily = _complete_daily_grid(scoped_observed_daily)
		weekly = _build_weekly_frame(daily)
		raw_context = self._build_raw_context(daily, items, stores, raw_stats)
		metadata = {
			"dataset_kind": self.kind,
			"source_frequency": "daily",
			"weekly_rule": self.manifest.week_anchor,
			"stockout_risk": "high",
			"structural_shift_risk": "moderate",
			"series_limit": self.manifest.series_limit,
			"series_count_after_sampling": int(weekly["series_id"].nunique()),
			"dataset_adapter": self.__class__.__name__,
			"business_category_definition": "Native Favorita items.family hierarchy",
			"target_proxy_type": "native_unit_sales",
			"target_proxy_status": "native",
			"raw_input_object": "favorita train + items + stores",
			"raw_context_label": "raw Favorita daily context",
		}
		policy_flags = {
			"target_proxy": False,
			"sku_supported": True,
			"zero_sales_missing_in_raw": True,
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
		daily: pd.DataFrame,
		items: pd.DataFrame,
		stores: pd.DataFrame,
		raw_stats: dict[str, Any],
	) -> dict[str, Any]:
		daily_total_sales = (
			daily.groupby("date", as_index=False)["sales_units"]
			.sum()
			.sort_values("date")
			.reset_index(drop=True)
		)
		item_counts_by_category = (
			items.assign(category=items["family"].fillna("unknown").astype(str))
			.groupby("category", as_index=False)["item_nbr"]
			.nunique()
			.rename(columns={"item_nbr": "item_count"})
		)
		store_sample = stores.head(10).reset_index(drop=True)
		daily_series_by_series_id = {
			str(series_id): series_daily.loc[:, ["date", "sales_units"]].sort_values("date").reset_index(drop=True)
			for series_id, series_daily in daily.groupby("series_id", sort=False)
		}

		return {
			"dataset_kind": self.kind,
			"dataset_adapter": self.__class__.__name__,
			"raw_input_object": "favorita train + items + stores",
			"raw_context_label": "raw Favorita daily context",
			"raw_item_count": int(raw_stats["raw_item_count"]),
			"raw_store_count": int(raw_stats["raw_store_count"]),
			"raw_state_count": int(stores["state"].nunique()) if "state" in stores.columns else None,
			"raw_department_count": int(raw_stats["raw_department_count"]),
			"raw_daily_observations": int(raw_stats["raw_daily_observations"]),
			"daily_total_sales": daily_total_sales,
			"item_counts_by_category": item_counts_by_category,
			"raw_sales_sample": raw_stats["raw_sales_sample"],
			"calendar_sample": daily_total_sales.head(10).reset_index(drop=True),
			"calendar_period_start": str(raw_stats["calendar_period_start"].date()),
			"calendar_period_end": str(raw_stats["calendar_period_end"].date()),
			"store_table_sample": store_sample,
			"negative_return_rows": int(raw_stats["negative_return_rows"]),
			"daily_series_by_series_id": daily_series_by_series_id,
			"raw_zero_sales_note": (
				"Favorita train omits many store-item-date combinations with zero sales; "
				"adapter rebuilds full daily grids per modeled series before weekly aggregation."
			),
		}


def _build_weekly_frame(daily: pd.DataFrame) -> pd.DataFrame:
	daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")

	group_columns = ["series_id", "category", "week_start"]
	if "sku" in daily.columns:
		group_columns.insert(2, "sku")

	weekly = (
		daily.groupby(group_columns, as_index=False)
		.agg(
			sales_units=("sales_units", "sum"),
			days_in_week=("date", "nunique"),
		)
		.sort_values(["series_id", "week_start"])
		.reset_index(drop=True)
	)
	return weekly


def _build_observed_daily_frame(
	data: DatasetLoadData,
	items: pd.DataFrame,
	aggregation_level: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
	item_categories = (
		items.assign(category=items["family"].fillna("unknown").astype(str))
		.set_index("item_nbr")["category"]
	)
	chunk_frames: list[pd.DataFrame] = []
	raw_item_ids: set[int] = set()
	raw_store_ids: set[int] = set()
	raw_dates: set[pd.Timestamp] = set()
	negative_return_rows = 0
	raw_sales_sample: pd.DataFrame | None = None
	calendar_period_start: pd.Timestamp | None = None
	calendar_period_end: pd.Timestamp | None = None

	for chunk in _iter_train_chunks(data):
		chunk = chunk.loc[:, _TRAIN_COLUMNS].copy()
		chunk["date"] = pd.to_datetime(chunk["date"], utc=False)
		chunk["category"] = chunk["item_nbr"].map(item_categories).fillna("unknown").astype(str)
		chunk["sales_units"] = chunk["unit_sales"].astype(float)
		negative_return_rows += int((chunk["sales_units"] < 0).sum())
		raw_item_ids.update(int(value) for value in chunk["item_nbr"].dropna().unique())
		raw_store_ids.update(int(value) for value in chunk["store_nbr"].dropna().unique())
		raw_dates.update(pd.Timestamp(value) for value in chunk["date"].dropna().unique())

		chunk_start = chunk["date"].min()
		chunk_end = chunk["date"].max()
		calendar_period_start = chunk_start if calendar_period_start is None else min(calendar_period_start, chunk_start)
		calendar_period_end = chunk_end if calendar_period_end is None else max(calendar_period_end, chunk_end)
		if raw_sales_sample is None:
			raw_sales_sample = (
				chunk.loc[:, ["date", "store_nbr", "item_nbr", "sales_units", "category"]]
				.head(10)
				.reset_index(drop=True)
			)

		chunk_frames.append(_aggregate_observed_daily_chunk(chunk, aggregation_level))

	if not chunk_frames:
		raise ValueError("Favorita train source is empty")

	observed_daily = (
		pd.concat(chunk_frames, ignore_index=True)
		.groupby(_daily_group_columns(aggregation_level), as_index=False)
		.agg(sales_units=("sales_units", "sum"))
		.sort_values(["series_id", "date"])
		.reset_index(drop=True)
	)
	if calendar_period_start is None or calendar_period_end is None:
		raise ValueError("Favorita train source does not contain valid dates")
	raw_stats = {
		"raw_item_count": len(raw_item_ids),
		"raw_store_count": len(raw_store_ids),
		"raw_department_count": int(items["family"].fillna("unknown").nunique()),
		"raw_daily_observations": len(raw_dates),
		"raw_sales_sample": raw_sales_sample if raw_sales_sample is not None else pd.DataFrame(),
		"calendar_period_start": calendar_period_start,
		"calendar_period_end": calendar_period_end,
		"negative_return_rows": negative_return_rows,
	}
	return observed_daily, raw_stats


def _iter_train_chunks(data: DatasetLoadData) -> Iterator[pd.DataFrame]:
	if "train" in data.tables and not data.tables["train"].empty:
		yield data.require_table("train").copy()
		return

	train_path = data.metadata.get("train_path")
	if not isinstance(train_path, str) or not train_path.strip():
		raise KeyError("train")

	yield from pd.read_csv(
		train_path,
		usecols=_TRAIN_COLUMNS,
		dtype=_TRAIN_DTYPES,
		chunksize=_TRAIN_CHUNK_SIZE,
	)


def _aggregate_observed_daily_chunk(chunk: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
	if aggregation_level == "category":
		return (
			chunk.groupby(["category", "date"], as_index=False)
			.agg(sales_units=("sales_units", "sum"))
			.assign(series_id=lambda frame: frame["category"])
			.loc[:, ["series_id", "category", "date", "sales_units"]]
		)
	elif aggregation_level == "sku":
		aggregated = (
			chunk.groupby(["item_nbr", "category", "date"], as_index=False)
			.agg(sales_units=("sales_units", "sum"))
			.rename(columns={"item_nbr": "sku"})
		)
		aggregated["sku"] = aggregated["sku"].astype(str)
		aggregated["series_id"] = aggregated["sku"]
		return aggregated.loc[:, ["series_id", "category", "sku", "date", "sales_units"]]

	raise ValueError(f"Unsupported aggregation level for Favorita: {aggregation_level}")


def _daily_group_columns(aggregation_level: str) -> list[str]:
	if aggregation_level == "category":
		return ["series_id", "category", "date"]
	if aggregation_level == "sku":
		return ["series_id", "category", "sku", "date"]
	raise ValueError(f"Unsupported aggregation level for Favorita: {aggregation_level}")


def _apply_series_scope(
	observed_daily: pd.DataFrame,
	*,
	series_allowlist: list[str] | None,
	series_limit: int | None,
) -> pd.DataFrame:
	scoped = observed_daily
	if series_allowlist is not None:
		scoped = _apply_series_allowlist(scoped, series_allowlist)
	if series_limit is not None:
		scoped = _apply_series_limit(scoped, series_limit)
	return scoped.reset_index(drop=True)


def _complete_daily_grid(observed_daily: pd.DataFrame) -> pd.DataFrame:
	group_columns = [column for column in ("series_id", "category", "sku") if column in observed_daily.columns]
	date_index = pd.date_range(start=observed_daily["date"].min(), end=observed_daily["date"].max(), freq="D")
	series_meta = observed_daily.loc[:, group_columns].drop_duplicates().reset_index(drop=True)
	series_meta["__join_key"] = 1
	calendar = pd.DataFrame({"date": date_index, "__join_key": 1})
	full_grid = series_meta.merge(calendar, on="__join_key", how="inner").drop(columns="__join_key")

	daily = full_grid.merge(
		observed_daily,
		on=[*group_columns, "date"],
		how="left",
	)
	daily["sales_units"] = daily["sales_units"].fillna(0.0)
	return daily.sort_values(["series_id", "date"]).reset_index(drop=True)


def _apply_series_limit(frame: pd.DataFrame, series_limit: int) -> pd.DataFrame:
	series_totals = (
		frame.groupby("series_id", as_index=False)
		.agg(total_sales_units=("sales_units", "sum"))
		.sort_values(["total_sales_units", "series_id"], ascending=[False, True])
	)
	selected_series = series_totals.head(series_limit)["series_id"]
	sort_columns = _series_sort_columns(frame)
	return (
		frame[frame["series_id"].isin(selected_series)]
		.sort_values(sort_columns)
		.reset_index(drop=True)
	)


def _apply_series_allowlist(frame: pd.DataFrame, series_allowlist: list[str]) -> pd.DataFrame:
	filtered = frame[frame["series_id"].astype(str).isin(series_allowlist)].copy()
	if filtered.empty:
		raise ValueError("series_allowlist did not match any Favorita series_id")
	return filtered.sort_values(_series_sort_columns(filtered)).reset_index(drop=True)


def _series_sort_columns(frame: pd.DataFrame) -> list[str]:
	sort_columns = ["series_id"]
	if "date" in frame.columns:
		sort_columns.append("date")
	elif "week_start" in frame.columns:
		sort_columns.append("week_start")
	return sort_columns
