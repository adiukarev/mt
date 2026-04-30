import pandas as pd


def resolve_calendar_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить календарные признаки, известные заранее"""

	data["week_of_year"] = data["week_start"].dt.isocalendar().week.astype(int)
	data["month"] = data["week_start"].dt.month.astype(int)
	data["quarter"] = data["week_start"].dt.quarter.astype(int)
	data["year"] = data["week_start"].dt.year.astype(int)
	data["week_in_month"] = ((data["week_start"].dt.day - 1) // 7 + 1).astype(int)

	feature_columns = ["week_of_year", "month", "quarter", "year", "week_in_month"]

	return data, feature_columns
