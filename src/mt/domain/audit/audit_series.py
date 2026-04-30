import numpy as np
import pandas as pd

OUTLIER_ROLLING_WINDOW = 28
OUTLIER_MODIFIED_Z_THRESHOLD = 3.5


def compute_outlier_share(series: pd.Series) -> float:
	mask = compute_outlier_mask(series)
	if mask.empty:
		return 0.0
	return float(mask.mean())


def compute_outlier_mask(series: pd.Series) -> pd.Series:
	values = series.dropna().astype(float)
	if len(values) <= 1:
		return pd.Series(False, index=values.index, dtype=bool)

	window = min(OUTLIER_ROLLING_WINDOW, len(values))
	half_window = window // 2
	mask = pd.Series(False, index=values.index, dtype=bool)

	for position, index in enumerate(values.index):
		start = max(0, position - half_window)
		end = min(len(values), start + window)
		start = max(0, end - window)
		window_values = values.iloc[start:end]
		if len(window_values) < 3:
			continue

		median_value = float(window_values.median())
		mad_value = float((window_values - median_value).abs().median())
		if mad_value == 0.0:
			continue

		modified_z_score = 0.6745 * (float(values.loc[index]) - median_value) / mad_value
		mask.loc[index] = abs(modified_z_score) > OUTLIER_MODIFIED_Z_THRESHOLD

	return mask


def compute_coefficient_of_variation(values: pd.Series) -> float:
	clean_values = values.dropna().astype(float)
	if clean_values.empty:
		return float("nan")
	mean_value = float(clean_values.mean())
	if mean_value == 0.0:
		return float("nan")
	return float(clean_values.std(ddof=0) / mean_value)


def compute_trend_strength(values: pd.Series) -> float:
	clean_values = values.dropna().astype(float).reset_index(drop=True)
	if len(clean_values) <= 1:
		return 0.0
	x = np.arange(len(clean_values), dtype=float)
	y = clean_values.to_numpy(dtype=float)
	if float(np.std(y, ddof=0)) == 0.0:
		return 0.0
	slope, intercept = np.polyfit(x, y, deg=1)
	y_pred = slope * x + intercept
	ss_tot = float(np.sum((y - y.mean()) ** 2))
	if ss_tot == 0.0:
		return 0.0
	ss_res = float(np.sum((y - y_pred) ** 2))
	r_squared = 1.0 - (ss_res / ss_tot)
	if np.isnan(r_squared):
		return 0.0
	return float(np.clip(r_squared, 0.0, 1.0))


def compute_skewness(values: pd.Series) -> float:
	clean_values = values.dropna().astype(float)
	if len(clean_values) <= 2 or float(clean_values.std(ddof=0)) == 0.0:
		return 0.0
	return float(clean_values.skew())


def compute_sales_scale(values: pd.Series) -> float:
	clean_values = values.dropna().astype(float)
	if clean_values.empty:
		return 0.0
	return float(clean_values.mean())


def compute_volatility(values: pd.Series) -> float:
	clean_values = values.dropna().astype(float)
	if clean_values.empty:
		return 0.0
	return float(clean_values.std(ddof=0))


def split_mean_levels(values: pd.Series) -> tuple[float, float]:
	first_half, second_half = split_series_in_halves(values)
	return (
		float(first_half.mean()) if len(first_half) else float("nan"),
		float(second_half.mean()) if len(second_half) else float("nan"),
	)


def split_std_levels(values: pd.Series) -> tuple[float, float]:
	first_half, second_half = split_series_in_halves(values)
	return (
		float(first_half.std(ddof=0)) if len(first_half) else float("nan"),
		float(second_half.std(ddof=0)) if len(second_half) else float("nan"),
	)


def split_series_in_halves(values: pd.Series) -> tuple[pd.Series, pd.Series]:
	clean_values = values.dropna().astype(float).reset_index(drop=True)
	if clean_values.empty:
		return clean_values, clean_values
	midpoint = max(len(clean_values) // 2, 1)
	return clean_values.iloc[:midpoint], clean_values.iloc[midpoint:]


def safe_ratio(numerator: float, denominator: float) -> float:
	if np.isnan(numerator) or np.isnan(denominator) or denominator == 0.0:
		return float("nan")
	return float(numerator / denominator)


def safe_autocorr(series: pd.Series, lag: int) -> float:
	clean_values = series.dropna().astype(float).reset_index(drop=True)
	if len(clean_values) <= lag:
		return float("nan")
	if float(clean_values.std(ddof=0)) == 0.0:
		return 0.0
	return float(clean_values.autocorr(lag=lag))


def infer_focus_lags(acf_lag_1: float, acf_lag_7: float, acf_lag_28: float, acf_lag_52: float) -> str:
	candidates = [
		("1", acf_lag_1),
		("7", acf_lag_7),
		("28", acf_lag_28),
		("52", acf_lag_52),
	]
	selected = [lag for lag, value in candidates if pd.notna(value) and value >= 0.20]
	if not selected:
		selected = [lag for lag, value in candidates if pd.notna(value) and value >= 0.05]
	return "[" + ", ".join(selected or ["1"]) + "]"
