import numpy as np
import pandas as pd


def build_weekly_sales_format(
	sales: pd.DataFrame,
	calendar: pd.DataFrame,
	group_columns: list[str],
	aggregation_level: str,
	target_name: str,
) -> pd.DataFrame:
	"""Агрегировать широкий sales_train_evaluation в недельный формат

	Вход:
	- широкий дневной формат M5:
	  `item_id | cat_id | d_1 | d_2 | d_3 | ...`

	Промежуточная идея:
	- через `calendar` понять, какие `d_*` относятся к одной неделе;
	- сложить эти дневные колонки внутри недели;
	- перевести широкую weekly-таблицу в long-format.

	Выход:
	- `series_id | category | week_start | sales_units`

	Мини-пример:
	- было: `FOODS | d_1=5 | d_2=3 | d_3=1`
	- если `d_1,d_2` относятся к неделе `2011-01-24`, а `d_3` к неделе `2011-01-31`,
	  то после агрегации получим:
	  `FOODS | 2011-01-24 | 8`
	  `FOODS | 2011-01-31 | 1`
	"""

	# Берем все дневные колонки продаж: d_1, d_2, ..., d_N.
	day_columns = [column for column in sales.columns if column.startswith("d_")]
	# Оставляем в календаре только те дни, которые реально присутствуют в sales.
	valid_calendar = calendar[calendar["d"].isin(day_columns)].copy()
	# Сначала агрегируем строки до требуемого уровня моделирования:
	# - category: суммируем все SKU внутри категории;
	# - sku: сохраняем отдельные item_id.
	#
	# Пример до groupby:
	# item_id      cat_id   d_1 d_2
	# FOODS_1_001  FOODS      1   2
	# FOODS_1_002  FOODS      4   1
	#
	# После groupby по category:
	# cat_id       d_1 d_2
	# FOODS          5   3
	grouped = sales.loc[:, [*group_columns, *day_columns]].groupby(group_columns, as_index=False)[
		day_columns].sum()
	# Строим словарь "неделя -> список дневных колонок этой недели".
	#
	# Пример:
	# {
	#   Timestamp('2011-01-24'): ['d_1', 'd_2'],
	#   Timestamp('2011-01-31'): ['d_3', 'd_4', 'd_5', 'd_6', 'd_7'],
	# }
	week_to_days = (
		valid_calendar.sort_values("date")
		.groupby("week_start", sort=True)["d"]
		.apply(list)
		.to_dict()
	)

	weekly_values: dict[str, np.ndarray] = {}
	for week_start, days in week_to_days.items():
		# Суммируем только те day-колонки, которые календарь относит к данной неделе.
		#
		# Если для одной серии:
		# d_1=5, d_2=3, d_3=1, d_4=2, d_5=4, d_6=0, d_7=2
		# и неделя 2011-01-31 состоит из d_3..d_7,
		# то weekly sales для нее будут 1+2+4+0+2 = 9.
		weekly_values[str(pd.Timestamp(week_start).date())] = grouped[days].sum(axis=1).to_numpy()

	# Получаем wide weekly table:
	# category | 2011-01-24 | 2011-01-31 | ...
	wide_weekly = pd.DataFrame(weekly_values)
	panel = pd.concat([grouped[group_columns].reset_index(drop=True), wide_weekly], axis=1)
	# Переводим wide weekly table в long-format panel:
	# category | week_start | sales_units
	weekly = panel.melt(
		id_vars=group_columns,
		value_vars=list(weekly_values),
		var_name="week_start",
		value_name=target_name,
	)
	weekly["week_start"] = pd.to_datetime(weekly["week_start"], utc=False)
	# Нормализуем имена колонок под единый downstream-контракт.
	weekly = weekly.rename(columns={"cat_id": "category", target_name: "sales_units"})
	if aggregation_level == "category":
		weekly["series_id"] = weekly["category"]
	else:
		# На SKU-уровне серия идентифицируется самим товаром.
		weekly = weekly.rename(columns={"item_id": "sku"})
		weekly["series_id"] = weekly["sku"]
	return weekly.sort_values(["series_id", "week_start"]).reset_index(drop=True)


def build_weekly_price_format(
	sales: pd.DataFrame,
	sell_prices: pd.DataFrame,
	calendar: pd.DataFrame,
	aggregation_level: str,
) -> pd.DataFrame:
	"""Построить weekly price covariate без зависимости от realized sales.

	M5 хранит цены по `wm_yr_wk`, а продажи после подготовки живут по `week_start`.
	Задача метода: аккуратно сопоставить эти два календаря и получить одну weekly price
	на ту же серию и ту же неделю, где лежат weekly sales.

	Мини-пример:
	- для недели `2011-01-31` 5 дней попали в retail-week с ценой 10,
	  а 2 дня в следующую retail-week с ценой 12;
	- тогда недельная цена считается не как простое среднее `(10 + 12) / 2`,
	  а как взвешенное по числу дней:
	  `(10*5 + 12*2) / 7 = 10.57`.
	"""

	day_columns = [column for column in sales.columns if column.startswith("d_")]
	item_metadata = sales[["item_id", "cat_id", "store_id"]].drop_duplicates().copy()
	calendar_price_map = (
		calendar.loc[calendar["d"].isin(day_columns), ["week_start", "wm_yr_wk"]]
		.value_counts()
		.rename("day_count")
		.reset_index()
	)
	price_panel = (
		sell_prices.merge(item_metadata, on=["store_id", "item_id"], how="inner")
		.merge(calendar_price_map, on="wm_yr_wk", how="inner")
	)
	price_panel["weighted_price"] = price_panel["sell_price"] * price_panel["day_count"]
	item_store_weekly_price = (
		price_panel.groupby(["item_id", "cat_id", "store_id", "week_start"], as_index=False)
		.agg(weighted_price=("weighted_price", "sum"), total_day_count=("day_count", "sum"))
	)
	item_store_weekly_price["price"] = (
		item_store_weekly_price["weighted_price"]
		/ item_store_weekly_price["total_day_count"].replace(0, np.nan)
	)

	if aggregation_level == "category":
		aggregated = (
			item_store_weekly_price.groupby(["cat_id", "week_start"], as_index=False)
			.agg(price=("price", "mean"))
		)
		aggregated = aggregated.rename(columns={"cat_id": "category"})
		aggregated["series_id"] = aggregated["category"]
		return aggregated.loc[:, ["series_id", "category", "week_start", "price"]]

	aggregated = (
		item_store_weekly_price.groupby(["item_id", "cat_id", "week_start"], as_index=False)
		.agg(price=("price", "mean"))
	)
	aggregated = aggregated.rename(columns={"item_id": "sku", "cat_id": "category"})
	aggregated["series_id"] = aggregated["sku"]
	return aggregated.loc[:, ["series_id", "category", "week_start", "price"]]
