import numpy as np
import pandas as pd

from mt.domain.dataset import DatasetBundle, DatasetLoadData
from mt.domain.manifest import DatasetManifest
from mt.infra.dataset.build_weekly import build_weekly_price_format, build_weekly_sales_format


def prepare_dataset(manifest: DatasetManifest, data: DatasetLoadData) -> DatasetBundle:
	"""Преобразовать M5 в унифицированный недельный формат.

	Из:
	- `cat_id | d_1 | d_2 | d_3 | ...`
	- `FOODS  |   5 |   3 |   1 | ...`

	Сделать это:
	- `series_id | category | week_start | sales_units`
	- `FOODS     | FOODS    | 2011-01-24 | 8`
	- `FOODS     | FOODS    | 2011-01-31 | 14`

	для последующего feature engineering, backtesting и обучения моделей.
	"""

	calendar = data.calendar.copy()
	# приведение даты к нужному формату
	calendar["date"] = pd.to_datetime(calendar["date"], utc=False)
	# Каждая календарная дата приводится к понедельнику ее недели / 2011-02-02 (среда) -> 2011-01-31 (понедельник)
	calendar["week_start"] = calendar["date"] - pd.to_timedelta(
		calendar["date"].dt.weekday,
		unit="D"
	)

	# category - одна серия = одна категория
	# sku  - одна серия = один item внутри категории
	group_columns = ["cat_id"] \
		if manifest.aggregation_level == "category" \
		else ["item_id", "cat_id"]

	# Может объединить с build_weekly_price_format
	weekly = build_weekly_sales_format(
		data.sales,
		calendar,
		group_columns,
		manifest.aggregation_level,
		manifest.target_name,
	)

	weekly = weekly.merge(
		build_weekly_price_format(
			data.sales,
			data.sell_prices,
			calendar,
			manifest.aggregation_level,
		),
		on=["series_id", "category", "week_start"],
		how="left",
	)

	# Пока сценарии без промо (поддержка единого формата)
	weekly["promo"] = pd.NA

	weekly = weekly \
		if manifest.series_limit is None \
		else apply_series_limit(weekly, manifest.series_limit)

	return DatasetBundle(
		aggregation_level=manifest.aggregation_level,
		target_name=manifest.target_name,
		weekly=weekly,
		metadata={
			"source_frequency": "daily",
			"weekly_rule": manifest.week_anchor,
			"promo_available": False,
			"stockout_risk": "high",
			"series_limit": manifest.series_limit,
			"series_count_after_sampling": int(weekly["series_id"].nunique()),
		},
	)


def apply_series_limit(weekly: pd.DataFrame, series_limit: int) -> pd.DataFrame:
	"""Ограничить число рядов детерминированным top-volume subset для smoke/sample режима.

	- считаем суммарные продажи по каждой серии;
	- берем top-N серий по объему.

	- `FOODS=10000`, `HOBBIES=3000`, `HOUSEHOLD=2000`
	- при `series_limit=2` останутся только `FOODS` и `HOBBIES`
	"""

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
