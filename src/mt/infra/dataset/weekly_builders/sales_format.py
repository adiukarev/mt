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
	week_day_counts = {
		str(pd.Timestamp(week_start).date()): len(days)
		for week_start, days in week_to_days.items()
	}

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
	weekly["days_in_week"] = weekly["week_start"].dt.strftime("%Y-%m-%d").map(week_day_counts).astype(int)
	# Нормализуем имена колонок под единый downstream-контракт.
	weekly = weekly.rename(columns={"cat_id": "category", target_name: "sales_units"})
	if aggregation_level == "category":
		weekly["series_id"] = weekly["category"]
	else:
		# На SKU-уровне серия идентифицируется самим товаром.
		weekly = weekly.rename(columns={"item_id": "sku"})
		weekly["series_id"] = weekly["sku"]
	return weekly.sort_values(["series_id", "week_start"]).reset_index(drop=True)
