from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mt.domain.manifest import DatasetManifest, FeatureManifest


@dataclass(slots=True)
class ModelInfo:
	"""Базовая инфа модели"""

	# Имя модели
	model_name: str
	# Семейство модели
	model_family: str


@dataclass(slots=True)
class ModelResult:
	"""Таблица прогнозирования и метаданные работы модели"""

	# Метаданные модели
	info: ModelInfo
	# Таблица прогнозов по окнам
	predictions: pd.DataFrame
	# Предупреждения по запуску
	warnings: list[str] = field(default_factory=list)
	# Суммарное время обучения
	train_time_seconds: float | None = None
	# Суммарное время инференса
	inference_time_seconds: float | None = None
	# Полное время выполнения модели внутри run_model
	wall_time_seconds: float | None = None
	# Колонки, использованные моделью
	used_feature_columns: list[str] = field(default_factory=list)


class ForecastModelAdapter(ABC):
	"""Базовый интерфейс адаптера модели прогноза"""

	def __init__(self, model_info: ModelInfo) -> None:
		"""Сохранить сопоставимые метаданные модели"""

		self._model_info = model_info

	def prepare_frame(self, supervised: pd.DataFrame) -> pd.DataFrame:
		"""Подготовить модельно-специфичное представление supervised-таблицы"""

		return supervised

	def resolve_feature_columns(
		self,
		prepared_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> list[str]:
		"""
			Вернуть подмножество колонок, нужное модели.
			По умолчанию оставляет исходный набор признаков без фильтрации.
		"""

		return feature_columns

	def select_predict_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
	) -> pd.DataFrame:
		"""
			Отфильтровать строки перед инференсом по правилам модели.
			Базовая реализация ничего не отбрасывает.
		"""

		return predict_frame

	def select_inference_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> pd.DataFrame:
		"""
			Отфильтровать строки для боевого инференса без известных future target.
			Базовая реализация ничего не отбрасывает.
		"""

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
		"""Обучить модель на одном окне backtesting"""
		...

	@abstractmethod
	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
	) -> np.ndarray:
		"""Построить прогноз на один горизонт для текущей точки прогноза"""
		...

	def get_model_info(self) -> ModelInfo:
		"""Вернуть метаданные модели"""

		return self._model_info


@dataclass(slots=True)
class BestModelArtifact:
	"""Сериализуемый артефакт финальной модели для боевого прогноза"""

	model_name: str
	dataset_manifest: DatasetManifest
	feature_manifest: FeatureManifest
	feature_columns: list[str]
	horizons: list[int]
	adapters_by_horizon: dict[int, ForecastModelAdapter]
	training_aggregation_level: str
	training_last_week_start: str
