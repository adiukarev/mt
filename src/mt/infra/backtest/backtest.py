from dataclasses import asdict

import pandas as pd

from mt.domain.backtest import BacktestWindow
from mt.domain.manifest import BacktestManifest


def build_backtest_windows(
	weekly: pd.DataFrame,
	aggregation_level: str,
	manifest: BacktestManifest,
	feature_set: str,
	seed: int,
) -> list[BacktestWindow]:
	"""Построить окна оценки для rolling-origin backtesting"""

	unique_weeks = sorted(pd.to_datetime(weekly["week_start"].unique()))
	if len(unique_weeks) < manifest.min_train_weeks + manifest.horizon_max:
		raise ValueError()

	candidate_origins = unique_weeks[
		manifest.min_train_weeks - 1: len(unique_weeks) - manifest.horizon_max]
	candidate_origins = candidate_origins[:: manifest.step_weeks]

	if manifest.max_windows:
		candidate_origins = candidate_origins[-manifest.max_windows:]

	windows: list[BacktestWindow] = []
	train_start = unique_weeks[0]

	for horizon in range(manifest.horizon_min, manifest.horizon_max + 1):
		min_origin_idx = manifest.min_train_weeks - 1 + horizon
		horizon_origins = candidate_origins
		horizon_origins = [origin for origin in horizon_origins if unique_weeks.index(origin) >= min_origin_idx]

		for origin in horizon_origins:
			origin_idx = unique_weeks.index(origin)
			# train_end отступает назад на горизонт h: модель не должна видеть недели,
			# которые позже станут целью прогноза для данного окна.
			# При этом минимальная эффективная длина train остается не меньше min_train_weeks
			# для любого горизонта.
			train_end = unique_weeks[origin_idx - horizon]
			test_start = unique_weeks[origin_idx + horizon]

			windows.append(
				BacktestWindow(
					aggregation_level=aggregation_level,
					feature_set=feature_set,
					train_start=train_start,
					train_end=train_end,
					forecast_origin=origin,
					horizon=horizon,
					test_start=test_start,
					test_end=test_start,
					seed=seed,
				)
			)

	return windows
