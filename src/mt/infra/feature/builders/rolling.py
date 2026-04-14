import numpy as np
import pandas as pd


def add_rolling_features(
	panel: pd.DataFrame,
	rolling_windows: list[int],
	lags: list[int]
) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить rolling-признаки только по сдвинутой истории"""

	feature_columns: list[str] = []
	grouped = panel.groupby("series_id", group_keys=False)

	for window in rolling_windows:
		shifted = grouped["sales_units"].transform(lambda s: s.shift(1))
		window_shifted = grouped["sales_units"].transform(lambda s: s.shift(window))

		column_map = {
			f"rolling_{window}_mean": grouped["sales_units"].transform(
				lambda s: s.shift(1).rolling(window=window, min_periods=window).mean()),
			f"rolling_{window}_median": grouped["sales_units"].transform(
				lambda s: s.shift(1).rolling(window=window, min_periods=window).median()),
			f"rolling_{window}_std": grouped["sales_units"].transform(
				lambda s: s.shift(1).rolling(window=window, min_periods=window).std()),
			f"rolling_{window}_min": grouped["sales_units"].transform(
				lambda s: s.shift(1).rolling(window=window, min_periods=window).min()),
			f"rolling_{window}_max": grouped["sales_units"].transform(
				lambda s: s.shift(1).rolling(window=window, min_periods=window).max()),
		}

		for column, values in column_map.items():
			panel[column] = values
			feature_columns.append(column)

		mean_col = f"rolling_{window}_mean"
		ratio_col = f"rolling_{window}_ratio_to_rolling_mean"
		trend_col = f"rolling_{window}_recent_trend_delta"
		panel[ratio_col] = shifted / panel[mean_col].replace(0.0, np.nan)
		panel[trend_col] = shifted - window_shifted
		feature_columns.extend([ratio_col, trend_col])

	return panel, feature_columns
