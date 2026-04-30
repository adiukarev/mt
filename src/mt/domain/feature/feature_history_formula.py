import numpy as np
import pandas as pd

from mt.domain.feature.feature_history_formula_catalog import build_lag_feature_specs
from mt.domain.feature.feature_history_formula_default import EXTENDED_ROLLING_WINDOWS
from mt.domain.feature.feature_history_formula_name import build_rolling_feature_name


def compute_lag_features(
	data: pd.DataFrame,
	group_column: str,
	target_column: str,
	lags: list[int],
) -> tuple[pd.DataFrame, list[str]]:
	feature_columns: list[str] = []
	grouped = data.groupby(group_column, group_keys=False)
	for spec in build_lag_feature_specs(lags):
		data[spec.name] = grouped[target_column].shift(spec.lag)
		feature_columns.append(spec.name)
	return data, feature_columns


def compute_rolling_features(
	data: pd.DataFrame,
	group_column: str,
	target_column: str,
	windows: list[int],
) -> tuple[pd.DataFrame, list[str]]:
	feature_columns: list[str] = []
	grouped = data.groupby(group_column, group_keys=False)

	for window in sorted(set(windows)):
		shifted = grouped[target_column].transform(lambda series: series.shift(1))
		rolling_mean = grouped[target_column].transform(
			lambda series: series.shift(1).rolling(window=window, min_periods=window).mean()
		)
		rolling_median = grouped[target_column].transform(
			lambda series: series.shift(1).rolling(window=window, min_periods=window).median()
		)
		rolling_mad = grouped[target_column].transform(
			lambda series: series.shift(1).rolling(window=window, min_periods=window).apply(
				_compute_mad,
				raw=False,
			)
		)
		rolling_iqr = grouped[target_column].transform(
			lambda series: series.shift(1).rolling(window=window, min_periods=window).apply(
				_compute_iqr,
				raw=False,
			)
		)
		robust_zscore = 0.6745 * (shifted - rolling_median) / rolling_mad.replace(0.0, np.nan)

		aggregations: dict[str, pd.Series] = {
			build_rolling_feature_name("mean", window): rolling_mean,
			build_rolling_feature_name("median", window): rolling_median,
			build_rolling_feature_name("mad", window): rolling_mad,
			build_rolling_feature_name("iqr", window): rolling_iqr,
			build_rolling_feature_name("robust_zscore", window): robust_zscore,
			build_rolling_feature_name("recent_outlier_flag", window): (
				robust_zscore.abs() > 3.5
			).astype("float"),
		}

		if window in EXTENDED_ROLLING_WINDOWS:
			aggregations[build_rolling_feature_name("std", window)] = grouped[target_column].transform(
				lambda series: series.shift(1).rolling(window=window, min_periods=window).std()
			)
			aggregations[build_rolling_feature_name("max", window)] = grouped[target_column].transform(
				lambda series: series.shift(1).rolling(window=window, min_periods=window).max()
			)
			aggregations[build_rolling_feature_name("min", window)] = grouped[target_column].transform(
				lambda series: series.shift(1).rolling(window=window, min_periods=window).min()
			)

		for column, values in aggregations.items():
			data[column] = values
			feature_columns.append(column)

	return data, feature_columns


def _compute_mad(window_values: pd.Series) -> float:
	values = window_values.dropna().astype(float)
	if values.empty:
		return float("nan")
	median_value = float(values.median())
	return float((values - median_value).abs().median())


def _compute_iqr(window_values: pd.Series) -> float:
	values = window_values.dropna().astype(float)
	if values.empty:
		return float("nan")
	return float(values.quantile(0.75) - values.quantile(0.25))
