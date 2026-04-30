import numpy as np
import pandas as pd

from mt.domain.probabilistic.probabilistic_settings import (
	DEFAULT_INTERVAL_LEVELS,
	DEFAULT_PROBABILISTIC_QUANTILES,
)
from mt.domain.probabilistic.probabilistic import (
	CANONICAL_PROBABILISTIC_COLUMNS,
	LOWER_BOUND_COLUMNS,
	ORDERED_INTERVAL_STACK_COLUMNS,
	PROBABILISTIC_VALUE_COLUMNS,
	QUANTILE_COLUMNS,
	ProbabilisticColumn,
	ProbabilisticSource,
	ProbabilisticStatus,
)

BASE_PREDICTION_COLUMNS: tuple[str, ...] = (
	"model_name",
	"model_family",
	"series_id",
	"category",
	"segment_label",
	"forecast_origin",
	"target_date",
	"horizon",
	"actual",
	"prediction",
)
CANONICAL_PREDICTION_COLUMNS: tuple[str, ...] = BASE_PREDICTION_COLUMNS + CANONICAL_PROBABILISTIC_COLUMNS


def build_empty_prediction_frame() -> pd.DataFrame:
	return pd.DataFrame(columns=list(CANONICAL_PREDICTION_COLUMNS))


def ensure_prediction_schema(frame: pd.DataFrame) -> pd.DataFrame:
	result = frame.copy()
	numeric_defaults = {column: np.nan for column in PROBABILISTIC_VALUE_COLUMNS}
	for column, default_value in numeric_defaults.items():
		if column not in result.columns:
			result[column] = default_value
	for column, default_value in (
		(ProbabilisticColumn.SOURCE, ProbabilisticSource.NONE),
		(ProbabilisticColumn.STATUS, ProbabilisticStatus.POINT_ONLY),
	):
		if column not in result.columns:
			result[column] = default_value
	return result.loc[:, [column for column in CANONICAL_PREDICTION_COLUMNS if column in result.columns]]


def finalize_prediction_frame(
	frame: pd.DataFrame,
	clip_lower_to_zero: bool = True,
) -> tuple[pd.DataFrame, bool, bool]:
	result = ensure_prediction_schema(frame)
	crossing_corrected = _enforce_monotonicity(result)
	_align_quantiles_with_intervals(result)
	clipped = _clip_lower_bounds(result) if clip_lower_to_zero else False
	return result, crossing_corrected, clipped


def probabilistic_quantile_column_map() -> dict[float, ProbabilisticColumn]:
	return {
		0.10: ProbabilisticColumn.Q10,
		0.50: ProbabilisticColumn.Q50,
		0.90: ProbabilisticColumn.Q90,
	}


def interval_column_map() -> dict[float, tuple[ProbabilisticColumn, ProbabilisticColumn]]:
	return {
		0.80: (ProbabilisticColumn.LO_80, ProbabilisticColumn.HI_80),
		0.95: (ProbabilisticColumn.LO_95, ProbabilisticColumn.HI_95),
	}


def quantile_aliases(
	quantiles: tuple[float, ...] = DEFAULT_PROBABILISTIC_QUANTILES,
) -> dict[float, ProbabilisticColumn]:
	aliases = probabilistic_quantile_column_map()
	return {quantile: aliases[quantile] for quantile in quantiles if quantile in aliases}


def interval_aliases(
	levels: tuple[float, ...] = DEFAULT_INTERVAL_LEVELS,
) -> dict[float, tuple[ProbabilisticColumn, ProbabilisticColumn]]:
	aliases = interval_column_map()
	return {level: aliases[level] for level in levels if level in aliases}


def has_complete_probabilistic_output(frame: pd.DataFrame) -> bool:
	if frame.empty:
		return False
	if not set(PROBABILISTIC_VALUE_COLUMNS).issubset(frame.columns):
		return False
	return bool(frame.loc[:, list(PROBABILISTIC_VALUE_COLUMNS)].notna().all(axis=1).all())


def _enforce_monotonicity(frame: pd.DataFrame) -> bool:
	if frame.empty:
		return False
	changed = False
	available_quantiles = [column for column in QUANTILE_COLUMNS if column in frame.columns]
	if available_quantiles:
		values = frame[available_quantiles].to_numpy(dtype=float, copy=True)
		mask = np.isfinite(values)
		for row_idx in range(values.shape[0]):
			row_mask = mask[row_idx]
			if row_mask.sum() < 2:
				continue
			row_values = values[row_idx, row_mask]
			new_values = np.sort(row_values)
			if not np.allclose(row_values, new_values, equal_nan=True):
				changed = True
			values[row_idx, row_mask] = new_values
		frame.loc[:, available_quantiles] = values

	if set(ORDERED_INTERVAL_STACK_COLUMNS).issubset(frame.columns):
		stack = frame.loc[:, list(ORDERED_INTERVAL_STACK_COLUMNS)].to_numpy(dtype=float, copy=True)
		mask = np.isfinite(stack)
		for row_idx in range(stack.shape[0]):
			row_mask = mask[row_idx]
			if row_mask.sum() < 2:
				continue
			row_values = stack[row_idx, row_mask]
			new_values = np.sort(row_values)
			if not np.allclose(row_values, new_values, equal_nan=True):
				changed = True
			stack[row_idx, row_mask] = new_values
		frame.loc[:, list(ORDERED_INTERVAL_STACK_COLUMNS)] = stack
	return changed


def _clip_lower_bounds(frame: pd.DataFrame) -> bool:
	clipped = False
	for column in LOWER_BOUND_COLUMNS:
		if column not in frame.columns:
			continue
		series = frame[column].astype(float)
		if bool((series < 0.0).any()):
			frame[column] = series.clip(lower=0.0)
			clipped = True
	return clipped


def _align_quantiles_with_intervals(frame: pd.DataFrame) -> None:
	if {ProbabilisticColumn.LO_80, ProbabilisticColumn.Q10}.issubset(frame.columns):
		lower = pd.concat(
			[
				pd.to_numeric(frame[ProbabilisticColumn.LO_80], errors="coerce"),
				pd.to_numeric(frame[ProbabilisticColumn.Q10], errors="coerce"),
			],
			axis=1,
		).min(axis=1, skipna=True)
		frame[ProbabilisticColumn.LO_80] = lower
		frame[ProbabilisticColumn.Q10] = lower
	if {ProbabilisticColumn.HI_80, ProbabilisticColumn.Q90}.issubset(frame.columns):
		upper = pd.concat(
			[
				pd.to_numeric(frame[ProbabilisticColumn.HI_80], errors="coerce"),
				pd.to_numeric(frame[ProbabilisticColumn.Q90], errors="coerce"),
			],
			axis=1,
		).max(axis=1, skipna=True)
		frame[ProbabilisticColumn.HI_80] = upper
		frame[ProbabilisticColumn.Q90] = upper
	if {"prediction", ProbabilisticColumn.Q50}.issubset(frame.columns):
		frame[ProbabilisticColumn.Q50] = frame["prediction"].where(
			frame["prediction"].notna(),
			frame[ProbabilisticColumn.Q50],
		)
