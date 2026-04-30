from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestWindow:
	"""Одно окно оценки при сдвигаемой точке прогноза"""

	# Дата происхождения прогноза
	forecast_origin: pd.Timestamp
	# Горизонт прогноза в неделях
	horizon: int
	# Начало обучающего интервала
	train_start: pd.Timestamp
	# Конец обучающего интервала
	train_end: pd.Timestamp
	# Начало тестового интервала
	test_start: pd.Timestamp
	# Конец тестового интервала
	test_end: pd.Timestamp
