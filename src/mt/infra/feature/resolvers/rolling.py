import numpy as np
import pandas as pd

from mt.domain.feature.feature_history_formula import compute_rolling_features
from mt.domain.feature.feature_history_formula_name import (
	build_rolling_feature_name,
)


def resolve_rolling_features(
	data: pd.DataFrame,
	rolling_windows: list[int],
) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить rolling-признаки только по сдвинутой истории"""

	data, feature_columns = compute_rolling_features(
		data=data,
		group_column="series_id",
		target_column="sales_units",
		windows=rolling_windows,
	)

	grouped = data.groupby("series_id", group_keys=False)
	shifted = grouped["sales_units"].transform(lambda series: series.shift(1))

	for window in rolling_windows:
		window_shifted = grouped["sales_units"].transform(lambda series: series.shift(window))
		mean_col = build_rolling_feature_name("mean", window)
		ratio_col = build_rolling_feature_name("ratio_to_mean", window)
		trend_col = build_rolling_feature_name("recent_trend_delta", window)
		data[ratio_col] = shifted / data[mean_col].replace(0.0, np.nan)
		data[trend_col] = shifted - window_shifted
		feature_columns.extend([ratio_col, trend_col])

	return data, feature_columns
