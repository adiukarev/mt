import pandas as pd


def add_calendar_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить календарные признаки, известные заранее"""

	panel["week_of_year"] = panel["week_start"].dt.isocalendar().week.astype(int)
	panel["month"] = panel["week_start"].dt.month.astype(int)
	panel["quarter"] = panel["week_start"].dt.quarter.astype(int)
	panel["year"] = panel["week_start"].dt.year.astype(int)
	panel["week_in_month"] = ((panel["week_start"].dt.day - 1) // 7 + 1).astype(int)
	feature_columns = ["week_of_year", "month", "quarter", "year", "week_in_month"]

	return panel, feature_columns
