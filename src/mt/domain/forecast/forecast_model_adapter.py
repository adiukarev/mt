from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from mt.domain.model.model_info import ModelInfo
from mt.domain.probabilistic.probabilistic_settings import DEFAULT_PROBABILISTIC_QUANTILES


class ForecastModelAdapter(ABC):
	"""Базовый интерфейс адаптера модели прогноза"""

	def __init__(self, model_info: ModelInfo) -> None:
		self._model_info = model_info

	def prepare_frame(self, supervised: pd.DataFrame) -> pd.DataFrame:
		return supervised

	def resolve_feature_columns(
		self,
		prepared_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> list[str]:
		return feature_columns

	def select_predict_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
	) -> pd.DataFrame:
		return predict_frame

	def select_inference_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> pd.DataFrame:
		return predict_frame

	@abstractmethod
	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int,
	) -> None:
		...

	@abstractmethod
	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
	) -> np.ndarray:
		...

	def supports_native_probabilistic(self) -> bool:
		return False

	def predict_quantiles(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		quantiles: tuple[float, ...] = DEFAULT_PROBABILISTIC_QUANTILES,
	) -> pd.DataFrame | None:
		return None

	def get_model_info(self) -> ModelInfo:
		return self._model_info
