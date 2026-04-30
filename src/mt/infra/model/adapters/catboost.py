import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import RegressorMixin

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.model.model_info import ModelInfo
from mt.domain.model.model_family import ModelFamily
from mt.domain.model.model_config_ml_manifest import ModelConfigCatBoostManifest
from mt.domain.model.model_name import ModelName


class CatBoostAdapter(ForecastModelAdapter):
	"""Глобальный адаптер регрессора CatBoost"""

	def __init__(self, model_config: ModelConfigCatBoostManifest) -> None:
		super().__init__(ModelInfo(model_name=ModelName.CATBOOST, model_family=ModelFamily.ML))

		self.model_config = model_config
		self.model: RegressorMixin | None = None

	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int, seed: int) -> None:
		"""Обучить CatBoost на валидных строках окна"""

		valid_train = train_frame.dropna(subset=feature_columns + [target_column])

		if valid_train.empty:
			self.model = None
			return

		self.model = CatBoostRegressor(
			depth=self.model_config.depth,
			iterations=self.model_config.iterations,
			learning_rate=self.model_config.learning_rate,
			l2_leaf_reg=self.model_config.l2_leaf_reg,
			loss_function=self.model_config.loss_function,
			random_seed=seed,
			thread_count=1,
			allow_writing_files=False,
			verbose=False,
		)

		self.model.fit(valid_train[feature_columns], valid_train[target_column])

	def select_predict_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str
	) -> pd.DataFrame:
		return predict_frame.dropna(subset=feature_columns + [target_column]).copy()

	def select_inference_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> pd.DataFrame:
		return predict_frame.dropna(subset=feature_columns).copy()

	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int
	) -> np.ndarray:
		if self.model is None or predict_frame.empty:
			return np.asarray([], dtype=float)

		return np.asarray(self.model.predict(predict_frame[feature_columns]), dtype=float)
