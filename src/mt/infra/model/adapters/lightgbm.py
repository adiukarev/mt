import numpy as np
import pandas as pd

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.model.model_info import ModelInfo
from mt.domain.model.model_family import ModelFamily
from mt.domain.model.model_config_ml_manifest import ModelConfigLightGBMManifest
from mt.domain.model.model_name import ModelName
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin


class LightGBMAdapter(ForecastModelAdapter):
	"""Глобальный адаптер регрессора LightGBM"""

	def __init__(self, model_config: ModelConfigLightGBMManifest) -> None:
		super().__init__(ModelInfo(model_name=ModelName.LIGHTGBM, model_family=ModelFamily.ML))

		self.model_config = model_config
		self.model: RegressorMixin | None = None

	def fit(
		self,
		train_frame:
		pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int
	) -> None:
		valid_train = train_frame.dropna(subset=feature_columns + [target_column])

		if valid_train.empty:
			self.model = None
			return

		self.model = LGBMRegressor(
			n_estimators=self.model_config.n_estimators,
			learning_rate=self.model_config.learning_rate,
			max_depth=self.model_config.max_depth,
			num_leaves=self.model_config.num_leaves,
			min_child_samples=self.model_config.min_child_samples,
			subsample=self.model_config.subsample,
			colsample_bytree=self.model_config.colsample_bytree,
			objective=self.model_config.objective,
			alpha=self.model_config.objective_alpha,
			tweedie_variance_power=self.model_config.tweedie_variance_power,
			random_state=seed,
			n_jobs=1,
			force_row_wise=True,
			verbosity=-1,
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
