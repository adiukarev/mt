import pandas as pd


def add_lag_features(panel: pd.DataFrame, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить лаговые признаки в supervised-таблицу"""

	feature_columns: list[str] = []
	grouped = panel.groupby("series_id", group_keys=False)

	for lag in lags:
		column = f"lag_{lag}"
		panel[column] = grouped["sales_units"].shift(lag)
		feature_columns.append(column)

	return panel, feature_columns
