import numpy as np
import pandas as pd

from mt.domain.metric.metric_name import PROBABILISTIC_METRIC_NAMES
from mt.domain.probabilistic.probabilistic import (
	PROBABILISTIC_VALUE_COLUMNS,
	ProbabilisticColumn,
	ProbabilisticStatus,
)


def calculate_probabilistic_metrics(frame: pd.DataFrame) -> dict[str, float]:
	available_frame = _select_available_probabilistic_rows(frame)
	if available_frame.empty:
		return {metric_name: float("nan") for metric_name in PROBABILISTIC_METRIC_NAMES}

	actual = available_frame["actual"].astype(float).to_numpy()
	return {
		"Pinball_q10": calculate_pinball(actual, available_frame.get(ProbabilisticColumn.Q10), 0.10),
		"Pinball_q50": calculate_pinball(actual, available_frame.get(ProbabilisticColumn.Q50), 0.50),
		"Pinball_q90": calculate_pinball(actual, available_frame.get(ProbabilisticColumn.Q90), 0.90),
		"MeanPinball": calculate_mean_pinball(actual, available_frame),
		"Coverage80": calculate_coverage(
			actual,
			available_frame.get(ProbabilisticColumn.LO_80),
			available_frame.get(ProbabilisticColumn.HI_80),
		),
		"Coverage95": calculate_coverage(
			actual,
			available_frame.get(ProbabilisticColumn.LO_95),
			available_frame.get(ProbabilisticColumn.HI_95),
		),
		"Width80": calculate_interval_width(
			available_frame.get(ProbabilisticColumn.LO_80),
			available_frame.get(ProbabilisticColumn.HI_80),
		),
		"Width95": calculate_interval_width(
			available_frame.get(ProbabilisticColumn.LO_95),
			available_frame.get(ProbabilisticColumn.HI_95),
		),
		"WIS": calculate_wis(available_frame),
		"PICP80": calculate_coverage(
			actual,
			available_frame.get(ProbabilisticColumn.LO_80),
			available_frame.get(ProbabilisticColumn.HI_80),
		),
		"PICP95": calculate_coverage(
			actual,
			available_frame.get(ProbabilisticColumn.LO_95),
			available_frame.get(ProbabilisticColumn.HI_95),
		),
	}


def calculate_pinball(
	actual: np.ndarray,
	predicted: pd.Series | np.ndarray | None,
	quantile: float,
) -> float:
	if predicted is None:
		return float("nan")
	predicted_array = pd.Series(predicted).astype(float).to_numpy()
	mask = np.isfinite(actual) & np.isfinite(predicted_array)
	if not mask.any():
		return float("nan")
	error = actual[mask] - predicted_array[mask]
	return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def calculate_mean_pinball(actual: np.ndarray, frame: pd.DataFrame) -> float:
	values = [
		calculate_pinball(actual, frame.get(ProbabilisticColumn.Q10), 0.10),
		calculate_pinball(actual, frame.get(ProbabilisticColumn.Q50), 0.50),
		calculate_pinball(actual, frame.get(ProbabilisticColumn.Q90), 0.90),
	]
	if not all(np.isfinite(value) for value in values):
		return float("nan")
	return float(np.mean(values))


def calculate_coverage(
	actual: np.ndarray,
	lower: pd.Series | np.ndarray | None,
	upper: pd.Series | np.ndarray | None,
) -> float:
	if lower is None or upper is None:
		return float("nan")
	lower_array = pd.Series(lower).astype(float).to_numpy()
	upper_array = pd.Series(upper).astype(float).to_numpy()
	mask = np.isfinite(actual) & np.isfinite(lower_array) & np.isfinite(upper_array)
	if not mask.any():
		return float("nan")
	inside = (actual[mask] >= lower_array[mask]) & (actual[mask] <= upper_array[mask])
	return float(np.mean(inside.astype(float)))


def calculate_interval_width(
	lower: pd.Series | np.ndarray | None,
	upper: pd.Series | np.ndarray | None,
) -> float:
	if lower is None or upper is None:
		return float("nan")
	lower_array = pd.Series(lower).astype(float).to_numpy()
	upper_array = pd.Series(upper).astype(float).to_numpy()
	mask = np.isfinite(lower_array) & np.isfinite(upper_array)
	if not mask.any():
		return float("nan")
	return float(np.mean(upper_array[mask] - lower_array[mask]))


def calculate_wis(frame: pd.DataFrame) -> float:
	required_columns = {
		ProbabilisticColumn.Q50,
		ProbabilisticColumn.LO_80,
		ProbabilisticColumn.HI_80,
		ProbabilisticColumn.LO_95,
		ProbabilisticColumn.HI_95,
	}
	if not required_columns.issubset(frame.columns):
		return float("nan")
	actual = frame["actual"].astype(float).to_numpy()
	median = pd.Series(frame[ProbabilisticColumn.Q50]).astype(float).to_numpy()
	lo_80 = pd.Series(frame[ProbabilisticColumn.LO_80]).astype(float).to_numpy()
	hi_80 = pd.Series(frame[ProbabilisticColumn.HI_80]).astype(float).to_numpy()
	lo_95 = pd.Series(frame[ProbabilisticColumn.LO_95]).astype(float).to_numpy()
	hi_95 = pd.Series(frame[ProbabilisticColumn.HI_95]).astype(float).to_numpy()
	interval_specs = ((0.20, lo_80, hi_80), (0.05, lo_95, hi_95))
	component_values: list[float] = []
	for row_idx, observed in enumerate(actual):
		if not np.isfinite(observed):
			continue
		if not all(
			np.isfinite(value)
			for value in (
				median[row_idx],
				lo_80[row_idx],
				hi_80[row_idx],
				lo_95[row_idx],
				hi_95[row_idx],
			)
		):
			continue
		row_components = [0.5 * abs(observed - median[row_idx])]
		for alpha, lower, upper in interval_specs:
			lo_value = lower[row_idx]
			hi_value = upper[row_idx]
			penalty_low = max(lo_value - observed, 0.0)
			penalty_high = max(observed - hi_value, 0.0)
			row_components.append(
				(alpha / 2.0)
				* (
					(hi_value - lo_value)
					+ (2.0 / alpha) * penalty_low
					+ (2.0 / alpha) * penalty_high
				)
			)
		component_values.append(sum(row_components) / 2.5)
	if not component_values:
		return float("nan")
	return float(np.mean(component_values))


def _select_available_probabilistic_rows(frame: pd.DataFrame) -> pd.DataFrame:
	if frame.empty:
		return frame
	if ProbabilisticColumn.STATUS not in frame.columns:
		return frame.iloc[0:0].copy()
	result = frame[
		frame[ProbabilisticColumn.STATUS].astype(str) == ProbabilisticStatus.AVAILABLE
	].copy()
	if result.empty:
		return result
	available_columns = [column for column in PROBABILISTIC_VALUE_COLUMNS if column in result.columns]
	if len(available_columns) != len(PROBABILISTIC_VALUE_COLUMNS):
		return result.iloc[0:0].copy()
	return result[available_columns].notna().all(axis=1).pipe(lambda mask: result[mask].copy())
