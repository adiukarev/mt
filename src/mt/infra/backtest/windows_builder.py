from dataclasses import asdict

import pandas as pd

from mt.domain.backtest.backtest import BacktestWindow
from mt.domain.backtest.backtest_manifest import BacktestManifest


def build_backtest_windows(
	manifest: BacktestManifest,
	weekly: pd.DataFrame,
) -> list[BacktestWindow]:
	"""Построить окна оценки для rolling-origin backtesting"""

	unique_weeks = sorted(pd.to_datetime(weekly["week_start"].unique()))
	holdout_tail_weeks = manifest.resolve_holdout_tail_weeks()
	if len(unique_weeks) < manifest.min_train_weeks + manifest.horizon_end + holdout_tail_weeks:
		raise ValueError()

	max_horizon = manifest.horizon_end
	min_origin_idx = manifest.min_train_weeks - 1
	if manifest.shared_origin_grid:
		min_origin_idx += max_horizon

	max_origin_idx_exclusive = len(unique_weeks) - holdout_tail_weeks - max_horizon
	candidate_origins = unique_weeks[min_origin_idx:max_origin_idx_exclusive]
	candidate_origins = candidate_origins[:: manifest.step_weeks]

	windows: list[BacktestWindow] = []
	train_start = unique_weeks[0]
	week_to_idx = {week: idx for idx, week in enumerate(unique_weeks)}

	for horizon in range(manifest.horizon_start, manifest.horizon_end + 1):
		if manifest.shared_origin_grid:
			horizon_origins = candidate_origins
		else:
			horizon_min_origin_idx = manifest.min_train_weeks - 1 + horizon
			horizon_origins = [
				origin for origin in candidate_origins if week_to_idx[origin] >= horizon_min_origin_idx
			]

		for origin in horizon_origins:
			origin_idx = week_to_idx[origin]
			# forecast_origin is the last available historical week for this window.
			train_end = origin
			test_start = unique_weeks[origin_idx + horizon]

			windows.append(
				BacktestWindow(
					forecast_origin=origin,
					horizon=horizon,
					train_start=train_start,
					train_end=train_end,
					test_start=test_start,
					test_end=test_start,
				)
			)

	return windows
