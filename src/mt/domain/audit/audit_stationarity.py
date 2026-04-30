import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def calculate_adf(series: pd.Series) -> tuple[float, float]:
	values = series.dropna().astype(float)
	if len(values) < 10 or float(values.std(ddof=0)) == 0.0:
		return float("nan"), float("nan")
	stat, pvalue, *_ = adfuller(values, autolag="AIC")
	return float(stat), float(pvalue)


def calculate_kpss(series: pd.Series) -> tuple[float, float, str]:
	values = series.dropna().astype(float)
	if len(values) < 10 or float(values.std(ddof=0)) == 0.0:
		return float("nan"), float("nan"), "inconclusive"
	with warnings.catch_warnings(record=True) as caught_warnings:
		warnings.simplefilter("always", InterpolationWarning)
		stat, pvalue, *_ = kpss(values, regression="c", nlags="auto")
	for warning in caught_warnings:
		if issubclass(warning.category, InterpolationWarning):
			message = str(warning.message)
			if "greater than" in message:
				return float(stat), 0.1, "clipped_high"
			if "smaller than" in message:
				return float(stat), 0.01, "clipped_low"
	return float(stat), float(pvalue), "exact"


def infer_stationarity(adf_pvalue: float, kpss_pvalue: float, kpss_pvalue_note: str = "exact") -> str:
	if np.isnan(adf_pvalue) or np.isnan(kpss_pvalue):
		return "inconclusive"
	if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
		if kpss_pvalue_note == "clipped_high":
			return "likely_stationary_kpss_clipped_high"
		return "likely_stationary"
	if adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
		if kpss_pvalue_note == "clipped_low":
			return "likely_non_stationary_kpss_clipped_low"
		return "likely_non_stationary"
	if kpss_pvalue_note == "clipped_low":
		return "mixed_signal_kpss_clipped_low"
	if kpss_pvalue_note == "clipped_high":
		return "mixed_signal_kpss_clipped_high"
	return "mixed_signal"
