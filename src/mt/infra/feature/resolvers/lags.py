import pandas as pd

from mt.domain.feature.feature_history_formula import compute_lag_features


def resolve_lag_features(data: pd.DataFrame, lags: list[int]) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить лаговые признаки в supervised-таблицу"""
	return compute_lag_features(
		data=data,
		group_column="series_id",
		target_column="sales_units",
		lags=lags,
	)
