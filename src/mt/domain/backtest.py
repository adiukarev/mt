from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestWindow:
	"""Одно окно оценки при сдвигаемой точке прогноза"""

	# Уровень агрегации рядов
	aggregation_level: str
	# Набор признаков
	feature_set: str
	# Начало обучающего интервала
	train_start: pd.Timestamp
	# Конец обучающего интервала
	train_end: pd.Timestamp
	# Дата происхождения прогноза
	forecast_origin: pd.Timestamp
	# Горизонт прогноза в неделях
	horizon: int
	# Начало тестового интервала
	test_start: pd.Timestamp
	# Конец тестового интервала
	test_end: pd.Timestamp
	# Зафиксированный seed запуска
	seed: int
