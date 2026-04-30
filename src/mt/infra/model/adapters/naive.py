import numpy as np
import pandas as pd

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.model.model_info import ModelInfo
from mt.domain.model.model_family import ModelFamily
from mt.domain.model.model_name import ModelName


class NaiveAdapter(ForecastModelAdapter):
	"""Простой baseline: прогноз равен последнему наблюдению ряда"""

	model_name = ModelName.NAIVE

	def __init__(self) -> None:
		super().__init__(ModelInfo(model_name=self.model_name, model_family=ModelFamily.BASELINE))

	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int
	) -> None:
		"""
		Naive не обучается.

		Модель детерминирована правилом `prediction_t = lag_1_t`,
		поэтому на этапе fit ничего оценивать не нужно.
		"""

		return None

	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int
	) -> np.ndarray:
		"""Вернуть последний наблюденный уровень ряда."""

		return predict_frame["lag_1"].astype(float).to_numpy()
